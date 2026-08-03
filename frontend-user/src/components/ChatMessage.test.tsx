import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RiskLevel } from "../types";
import { ChatMessageView } from "./ChatMessage";

describe("consultation risk presentation", () => {
  it.each<[RiskLevel, string]>([
    ["low", "风险较低"],
    ["medium", "需要留意"],
    ["high", "风险较高"],
    ["emergency", "急症警示"]
  ])("renders %s risk with a text label", (riskLevel, label) => {
    render(
      <ChatMessageView
        message={{
          id: `message-${riskLevel}`,
          role: "assistant",
          content: "示例建议",
          payload: {
            riskLevel,
            suggestions: [],
            disclaimer: "",
            participants: [],
            safetyChecked: true,
            failed: false
          }
        }}
      />
    );

    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("does not render a trailing structured payload as answer content", () => {
    const { container } = render(
      <ChatMessageView
        message={{
          id: "message-structured-suffix",
          role: "assistant",
          content: [
            "## 居家建议",
            "请注意休息并持续观察。",
            "```json",
            '{"suggestions":["每4~6小时监测体温"],"risk_level":"medium"}',
            "```"
          ].join("\n"),
          payload: {
            riskLevel: "medium",
            suggestions: ["每4~6小时监测体温"],
            disclaimer: "以上信息仅供参考。",
            participants: ["健康咨询"],
            safetyChecked: true,
            failed: false
          }
        }}
      />
    );

    expect(screen.getByText("请注意休息并持续观察。")).toBeInTheDocument();
    expect(screen.getByText("每4~6小时监测体温", { selector: "li" })).toBeInTheDocument();
    expect(container.querySelector("pre")).not.toBeInTheDocument();
    expect(screen.queryByText(/"suggestions"/)).not.toBeInTheDocument();
  });
});
