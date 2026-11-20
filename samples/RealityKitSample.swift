// RealityKitSample.swift
// DetectBodyInVideo 导出 JSON 的 iOS / macOS 消费示例（任务 7.4）
//
// 用法：
// 1. 把导出的 `*_posefull.json` 拖进 Xcode 工程（务必勾选 target membership）。
// 2. USDZ 人形骨架的关节节点名必须与 JSON 的 `jointNames` 一致
//    （root / hips_joint / left_arm_joint ...，ARKit 命名）。
// 3. 在 RealityKit 场景里挂上 `PosePlaybackController`，每帧调用 `update(deltaTime:)`。
//
// 关键约定（与 schemaVersion 1.0.0 对齐）：
// - 单位：米；坐标系：右手系、y 向上、-z 向前。
//   加载后先断言 `coordinateSystem == "arkit-right-handed-y-up"`，不一致必须做转换再播放。
// - `joints[name]` 是**相对 root 的位移**（除 world 模式），`rootMotion.frames` 才是世界位移。
// - `valid == false` 的帧不要插值，直接保持上一姿态，避免模型抽搐。

import Foundation
import RealityKit
import SwiftUI

// MARK: - Schema

struct PoseDocument: Decodable {
    struct Source: Decodable {
        let file: String
        let fps: Double
        let width: Int
        let height: Int
        let frames: Int
        let durationSeconds: Double?
    }

    struct Joint: Decodable {
        let x: Double
        let y: Double
        let z: Double
        let visibility: Double

        var simd: SIMD3<Float> { SIMD3(Float(x), Float(y), Float(z)) }
    }

    struct Frame: Decodable {
        let index: Int
        let t: Double
        let valid: Bool
        let interpolated: Bool?
        let joints: [String: Joint]
        let angles: [String: Double]?
    }

    struct Quality: Decodable {
        let validFrameRatio: Double
        let meanVisibility: Double
        let verdict: String
        let verdictLabel: String?
    }

    struct RootMotionFrame: Decodable {
        struct Translation: Decodable {
            let x: Double
            let y: Double
            let z: Double
            var simd: SIMD3<Float> { SIMD3(Float(x), Float(y), Float(z)) }
        }

        struct Rotation: Decodable {
            let x: Double
            let y: Double
            let z: Double
            let w: Double
            var simd: simd_quatf {
                simd_quatf(ix: Float(x), iy: Float(y), iz: Float(z), r: Float(w))
            }
        }

        let index: Int
        let t: Double
        let translation: Translation
        let rotation: Rotation
    }

    struct RootMotion: Decodable {
        let mode: String
        let totalTranslation: Double
        let convention: String?
        let frames: [RootMotionFrame]
    }

    let version: String
    let generator: String
    let source: Source
    let coordinateSystem: String
    let units: String
    let personHeightMeters: Double
    let jointNames: [String]
    let frames: [Frame]
    let quality: Quality
    let rootMotion: RootMotion?
}

enum PoseDocumentError: LocalizedError {
    case unsupportedCoordinateSystem(String)
    case unsupportedUnits(String)
    case emptyFrames

    var errorDescription: String? {
        switch self {
        case .unsupportedCoordinateSystem(let value):
            return "坐标系 \(value) 与 RealityKit（y-up / 右手系 / -z 向前）不一致，请先转换"
        case .unsupportedUnits(let value):
            return "单位 \(value) 不是米，RealityKit 模型会比例错误"
        case .emptyFrames:
            return "JSON 没有任何帧，无法播放"
        }
    }
}

// MARK: - 播放控制器

@MainActor
final class PosePlaybackController: ObservableObject {
    @Published private(set) var elapsed: Double = 0
    @Published private(set) var isPlaying = false

    private let document: PoseDocument
    private let jointEntities: [String: Entity]
    private weak var rootEntity: Entity?
    private var cursor = 0

    init(document: PoseDocument, skeletonRoot: Entity) throws {
        guard document.coordinateSystem == "arkit-right-handed-y-up" else {
            throw PoseDocumentError.unsupportedCoordinateSystem(document.coordinateSystem)
        }
        guard document.units == "meters" else {
            throw PoseDocumentError.unsupportedUnits(document.units)
        }
        guard !document.frames.isEmpty else { throw PoseDocumentError.emptyFrames }

        self.document = document
        self.rootEntity = skeletonRoot

        var lookup: [String: Entity] = [:]
        for name in document.jointNames {
            if let entity = skeletonRoot.findEntity(named: name) {
                lookup[name] = entity
            }
        }
        self.jointEntities = lookup
    }

    var duration: Double { document.frames.last?.t ?? 0 }
    var qualityHint: String {
        "\(document.quality.verdictLabel ?? document.quality.verdict) · 有效帧 \(Int(document.quality.validFrameRatio * 100))%"
    }

    func play() { isPlaying = true }
    func pause() { isPlaying = false }

    /// 每帧调用；`deltaTime` 来自 `Scene.update` 或 SwiftUI TimelineView。
    func update(deltaTime: Double) {
        if isPlaying {
            elapsed = min(duration, elapsed + deltaTime)
            if elapsed >= duration { isPlaying = false }
        }
        applyPose(at: elapsed)
    }

    func seek(to time: Double) {
        elapsed = min(max(0, time), duration)
        applyPose(at: elapsed)
    }

    // MARK: 插值

    private func applyPose(at time: Double) {
        let frames = document.frames
        while cursor + 1 < frames.count, frames[cursor + 1].t <= time { cursor += 1 }
        while cursor > 0, frames[cursor].t > time { cursor -= 1 }

        let current = frames[cursor]
        let next = cursor + 1 < frames.count ? frames[cursor + 1] : nil

        // 无效帧：保持上一姿态，避免把"缺失"插值成乱抖
        guard current.valid else { return }

        var joints = current.joints
        var blend: Float = 0
        if let next, next.valid, next.t > current.t {
            blend = Float((time - current.t) / (next.t - current.t))
            for (name, value) in next.joints where joints[name] != nil {
                joints[name] = interpolate(joints[name]!, value, blend)
            }
        }

        for (name, entity) in jointEntities {
            guard let joint = joints[name] else { continue }
            let target = joint.simd
            if blend > 0.001 {
                entity.transform.translation = target
            } else {
                // 少量阻尼让 30fps 数据在 60/120Hz 屏幕上更稳
                var transform = entity.transform
                transform.translation = simd_mix(transform.translation, target, SIMD3(repeating: 0.55))
                entity.transform = transform
            }
        }

        applyRootMotion(at: time)
    }

    private func interpolate(_ a: PoseDocument.Joint, _ b: PoseDocument.Joint, _ t: Float) -> PoseDocument.Joint {
        PoseDocument.Joint(
            x: a.x + (b.x - a.x) * Double(t),
            y: a.y + (b.y - a.y) * Double(t),
            z: a.z + (b.z - a.z) * Double(t),
            visibility: min(a.visibility, b.visibility)
        )
    }

    private func applyRootMotion(at time: Double) {
        guard let root = rootEntity,
              let motion = document.rootMotion,
              !motion.frames.isEmpty else { return }

        let frames = motion.frames
        let index = frames.lastIndex { $0.t <= time } ?? 0
        let current = frames[index]
        root.transform.translation = current.translation.simd
        root.transform.rotation = current.rotation.simd
    }
}

// MARK: - SwiftUI 接入示例

struct PosePlayerView: View {
    let modelURL: URL
    let documentURL: URL

    @State private var controller: PosePlaybackController?
    @State private var error: String?

    var body: some View {
        RealityView { content in
            do {
                let model = try await Entity(contentsOf: modelURL)
                content.add(model)

                let data = try Data(contentsOf: documentURL)
                let document = try JSONDecoder().decode(PoseDocument.self, from: data)
                controller = try PosePlaybackController(document: document, skeletonRoot: model)
            } catch {
                self.error = error.localizedDescription
            }
        } update: { _ in
            controller?.update(deltaTime: 1.0 / 60.0)
        }
        .overlay(alignment: .bottom) {
            VStack(spacing: 8) {
                if let error {
                    Text(error).font(.footnote).foregroundStyle(.red)
                }
                if let controller {
                    Text(controller.qualityHint).font(.caption2).foregroundStyle(.secondary)
                    Slider(
                        value: Binding(
                            get: { controller.elapsed },
                            set: { controller.seek(to: $0) }
                        ),
                        in: 0...max(0.001, controller.duration)
                    )
                }
            }
            .padding()
        }
    }
}

// MARK: - 用 Scene.update 驱动（非 SwiftUI 场景）

/// 若使用 ARView / 自定义 Scene，把控制器挂到协调器上：
/// ```swift
/// func update(_ scene: Scene, deltaTime: TimeInterval) {
///     controller?.update(deltaTime: deltaTime)
/// }
/// ```
