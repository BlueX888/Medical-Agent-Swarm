import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConsultationSnapshot } from "../types";
import { LiveAnalysisCard } from "./LiveAnalysisCard";

describe("live analysis card", () => {
  it("explains the current patient-visible step while the answer is pending", () => {
    const snapshot: ConsultationSnapshot = {
      consultation_id: "consultation-live",
      status: "running",
      progress: {
        current_phase: "consulting",
        completed_phases: ["understanding", "planning"],
        participants: [],
        analysis_steps: [
          {
            id: "risk",
            label: "风险预检",
            summary: "当前信息未触发高风险路径，仍会保留症状加重时的就医提醒。",
            state: "done"
          },
          {
            id: "evidence",
            label: "资料核对",
            summary: "正在核对本地医学资料。",
            state: "active"
          },
          {
            id: "safety",
            label: "安全复核",
            summary: "回答生成后将进行复核。",
            state: "pending"
          }
        ],
        safety_checked: false
      },
      result: null,
      failure: null
    };

    render(<LiveAnalysisCard snapshot={snapshot} />);

    expect(screen.getByRole("heading", { name: "分析正在进行" })).toBeInTheDocument();
    expect(screen.getByText("资料核对")).toBeInTheDocument();
    expect(screen.getByText("正在核对本地医学资料。")).toBeInTheDocument();
    expect(screen.getByText("1 项已确认")).toBeInTheDocument();
    expect(screen.getByText("展示结构化进度，不展示模型内部思维记录")).toBeInTheDocument();
  });

  it("keeps a completed warning separate from the current pending step", () => {
    const snapshot: ConsultationSnapshot = {
      consultation_id: "consultation-warning",
      status: "running",
      progress: {
        current_phase: "safety_review",
        completed_phases: ["understanding", "planning", "consulting"],
        participants: [],
        analysis_steps: [
          {
            id: "risk",
            label: "风险预检",
            summary: "已识别需要重点留意的风险因素。",
            state: "attention"
          },
          {
            id: "safety",
            label: "安全复核",
            summary: "正在等待最终安全复核。",
            state: "pending"
          }
        ],
        safety_checked: false
      },
      result: null,
      failure: null
    };

    render(<LiveAnalysisCard snapshot={snapshot} />);

    expect(screen.getByText("安全复核")).toBeInTheDocument();
    expect(screen.getByText("正在等待最终安全复核。")).toBeInTheDocument();
    expect(screen.queryByText("已识别需要重点留意的风险因素。")).not.toBeInTheDocument();
  });
});
