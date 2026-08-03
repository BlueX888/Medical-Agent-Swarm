import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const api = vi.hoisted(() => ({
  getHealth: vi.fn(),
  getSessionMemory: vi.fn(),
  createConsultation: vi.fn(),
  getConsultation: vi.fn()
}));

vi.mock("./api/client", () => api);

describe("health consultation entry", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.setItem("mas-user-session", "session-test");
    api.getHealth.mockResolvedValue({ status: "ok", memory: { backend: "redis", status: "ok" } });
    api.getSessionMemory.mockResolvedValue({
      session_id: "session-test",
      backend: "redis",
      recent_history: []
    });
  });

  it("fills an example into the focused composer without sending it", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("说说最担心的症状")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "最近两天头痛并伴有恶心，需要马上就医吗？" }));

    const composer = screen.getByRole("textbox", { name: "描述你的症状或健康问题" });
    expect(composer).toHaveValue("最近两天头痛并伴有恶心，需要马上就医吗？");
    expect(composer).toHaveFocus();
    expect(api.createConsultation).not.toHaveBeenCalled();
  });

  it("shows the sanitized consultation result and safety status", async () => {
    const user = userEvent.setup();
    api.createConsultation.mockResolvedValue({ consultation_id: "consultation-1", status: "queued" });
    api.getConsultation.mockResolvedValue({
      consultation_id: "consultation-1",
      status: "success",
      progress: {
        current_phase: "finalizing",
        completed_phases: ["understanding", "planning", "consulting", "safety_review", "finalizing"],
        participants: [
          { id: "symptom_analysis", label: "风险与症状分析", state: "done" }
        ],
        safety_checked: true
      },
      result: {
        answer: "请立即联系急救服务。",
        risk_level: "emergency",
        suggestions: ["立即拨打 120"],
        disclaimer: "以上信息不能替代医生诊断。",
        participants: ["风险与症状分析"]
      },
      failure: null
    });
    render(<App />);

    const composer = await screen.findByRole("textbox", { name: "描述你的症状或健康问题" });
    await waitFor(() => expect(composer).toBeEnabled());
    await user.type(composer, "胸痛并且呼吸困难");
    await user.click(screen.getByRole("button", { name: "开始会诊" }));

    expect(await screen.findByText("请立即联系急救服务。")).toBeInTheDocument();
    expect(screen.getByText("急症警示")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("立即拨打 120");
    expect(screen.getAllByText("已完成安全复核").length).toBeGreaterThan(0);
    expect(api.createConsultation).toHaveBeenCalledWith({
      question: "胸痛并且呼吸困难",
      context: {},
      session_id: "session-test"
    });
    expect(api.getConsultation).toHaveBeenCalledWith("consultation-1", "session-test");
  });

  it("shows running roles and stops polling after a terminal snapshot", async () => {
    const user = userEvent.setup();
    api.createConsultation.mockResolvedValue({ consultation_id: "consultation-progress", status: "queued" });
    api.getConsultation
      .mockResolvedValueOnce({
        consultation_id: "consultation-progress",
        status: "running",
        progress: {
          current_phase: "consulting",
          completed_phases: ["understanding", "planning"],
          participants: [
            { id: "health_consultation", label: "健康咨询", state: "active" },
            { id: "evidence_research", label: "医学证据检索", state: "active" }
          ],
          safety_checked: false
        },
        result: null,
        failure: null
      })
      .mockResolvedValueOnce({
        consultation_id: "consultation-progress",
        status: "success",
        progress: {
          current_phase: "finalizing",
          completed_phases: ["understanding", "planning", "consulting", "safety_review", "finalizing"],
          participants: [
            { id: "health_consultation", label: "健康咨询", state: "done" },
            { id: "evidence_research", label: "医学证据检索", state: "done" }
          ],
          safety_checked: true
        },
        result: {
          answer: "分析已经完成。",
          risk_level: "low",
          suggestions: [],
          disclaimer: "",
          participants: ["健康咨询", "医学证据检索"]
        },
        failure: null
      });
    render(<App />);

    const composer = await screen.findByRole("textbox", { name: "描述你的症状或健康问题" });
    await waitFor(() => expect(composer).toBeEnabled());
    await user.type(composer, "持续头晕两天");
    await user.click(screen.getByRole("button", { name: "开始会诊" }));

    expect((await screen.findAllByText("分析中")).length).toBeGreaterThan(0);
    expect(await screen.findByText("分析已经完成。")).toBeInTheDocument();
    await new Promise((resolve) => window.setTimeout(resolve, 35));
    expect(api.getConsultation).toHaveBeenCalledTimes(2);
  });

  it("retries a failed consultation without duplicating the user message", async () => {
    const user = userEvent.setup();
    api.createConsultation
      .mockResolvedValueOnce({ consultation_id: "consultation-failed", status: "queued" })
      .mockResolvedValueOnce({ consultation_id: "consultation-retry", status: "queued" });
    api.getConsultation
      .mockResolvedValueOnce({
        consultation_id: "consultation-failed",
        status: "failed",
        progress: {
          current_phase: "finalizing",
          completed_phases: ["understanding", "planning", "consulting", "safety_review", "finalizing"],
          participants: [],
          safety_checked: false
        },
        result: null,
        failure: { code: "analysis_failed", message: "本轮分析未完成。", retryable: true }
      })
      .mockResolvedValueOnce({
        consultation_id: "consultation-retry",
        status: "success",
        progress: {
          current_phase: "finalizing",
          completed_phases: ["understanding", "planning", "consulting", "safety_review", "finalizing"],
          participants: [],
          safety_checked: true
        },
        result: {
          answer: "重新分析已完成。",
          risk_level: "low",
          suggestions: [],
          disclaimer: "",
          participants: []
        },
        failure: null
      });
    render(<App />);

    const composer = await screen.findByRole("textbox", { name: "描述你的症状或健康问题" });
    await waitFor(() => expect(composer).toBeEnabled());
    await user.type(composer, "持续头痛");
    await user.click(screen.getByRole("button", { name: "开始会诊" }));
    await user.click(await screen.findByRole("button", { name: "重新分析" }));

    expect(await screen.findByText("重新分析已完成。")).toBeInTheDocument();
    expect(screen.getAllByLabelText("你的问题")).toHaveLength(1);
    expect(api.createConsultation).toHaveBeenCalledTimes(2);
  });

  it("validates age in the health profile drawer", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("说说最担心的症状");

    const profileTrigger = screen.getByRole("button", { name: /健康资料/ });
    await user.click(profileTrigger);
    expect(screen.getByRole("button", { name: "关闭健康资料" })).toHaveFocus();
    const age = screen.getByRole("spinbutton", { name: "年龄" });
    await user.type(age, "121");

    expect(screen.getByText("请输入 0–120 的整数")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "完成" })).toBeDisabled();
    await user.clear(age);
    await user.type(age, "35");
    const done = screen.getByRole("button", { name: "完成" });
    expect(done).toBeEnabled();
    await user.click(done);
    expect(profileTrigger).toHaveFocus();
  });

  it("does not claim a restored answer passed safety review without persisted proof", async () => {
    api.getSessionMemory.mockResolvedValue({
      session_id: "session-test",
      backend: "redis",
      recent_history: [
        { role: "user", content: "之前的问题" },
        {
          role: "assistant",
          content: "之前的回答",
          metadata: {
            risk_level: "low",
            agents_involved: ["consultation_agent"],
            safety_checked: false
          }
        }
      ]
    });
    render(<App />);

    expect(await screen.findByText("之前的回答")).toBeInTheDocument();
    expect(screen.queryByText("已完成安全复核")).not.toBeInTheDocument();
  });

  it("starts a new conversation and offers an immediate return", async () => {
    const user = userEvent.setup();
    api.getSessionMemory.mockImplementation((sessionId: string) =>
      Promise.resolve({
        session_id: sessionId,
        backend: "redis",
        recent_history:
          sessionId === "session-test"
            ? [{ role: "user", content: "之前的问题" }, { role: "assistant", content: "之前的回答" }]
            : []
      })
    );
    render(<App />);
    expect(await screen.findByText("之前的问题")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "新对话" }));
    await user.click(screen.getByRole("button", { name: "开始新对话" }));

    const restore = await screen.findByRole("button", { name: "返回上一对话" });
    await user.click(restore);
    expect(await screen.findByText("之前的回答")).toBeInTheDocument();
  });
});
