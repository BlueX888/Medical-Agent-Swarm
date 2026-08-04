import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConsultationSnapshot } from "../types";
import { ProgressCard } from "./ProgressCard";

describe("patient-facing analysis progress", () => {
  it("shows a structured analysis receipt without presenting private reasoning", () => {
    const snapshot: ConsultationSnapshot = {
      consultation_id: "consultation-analysis",
      status: "running",
      progress: {
        current_phase: "consulting",
        completed_phases: ["understanding", "planning"],
        participants: [
          { id: "health_consultation", label: "健康咨询", state: "active" },
          { id: "evidence_research", label: "医学证据检索", state: "active" }
        ],
        analysis_steps: [
          {
            id: "risk",
            label: "风险预检",
            summary: "当前信息未触发高风险路径，仍会保留症状加重时的就医提醒。",
            state: "done"
          },
          {
            id: "focus",
            label: "本次重点",
            summary: "本次重点：可执行的生活调整、风险与就医时机。",
            state: "done"
          },
          {
            id: "evidence",
            label: "资料核对",
            summary: "已核对 3 条本地医学资料，并保留可引用来源。",
            state: "done"
          },
          {
            id: "collaboration",
            label: "协作分工",
            summary: "已安排 2 个分析角色：健康咨询、医学证据检索。",
            state: "active"
          },
          {
            id: "safety",
            label: "安全复核",
            summary: "回答生成后将检查急症提醒、过度诊断和用药风险。",
            state: "pending"
          }
        ],
        safety_checked: false
      },
      result: null,
      failure: null
    };

    render(<ProgressCard snapshot={snapshot} />);

    expect(screen.getByRole("heading", { name: "分析进度" })).toBeInTheDocument();
    expect(screen.getByText("本次分析摘要")).toBeInTheDocument();
    expect(screen.getByText("已核对 3 条本地医学资料，并保留可引用来源。")).toBeInTheDocument();
    expect(screen.getByText("这里展示可核验的步骤与结论，不包含模型内部思维记录。")).toBeInTheDocument();
    expect(screen.queryByText(/private|chain of thought/i)).not.toBeInTheDocument();
  });
});
