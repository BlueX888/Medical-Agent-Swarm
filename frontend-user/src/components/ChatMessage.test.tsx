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

  it("renders structured knowledge sources with safe external links", () => {
    render(
      <ChatMessageView
        message={{
          id: "message-with-source",
          role: "assistant",
          content: "慢性肾病管理需要个体化 [K1]。",
          payload: {
            riskLevel: "low",
            suggestions: [],
            disclaimer: "以上信息仅供参考。",
            participants: ["健康咨询"],
            sources: [
              {
                citation_id: "K1",
                title: "慢性肾病指南",
                source_org: "示例医学会",
                version: "2026",
                published_at: "2026-01-01",
                section: "治疗目标",
                external_url: "https://example.test/ckd"
              }
            ],
            safetyChecked: true,
            failed: false
          }
        }}
      />
    );

    expect(screen.getByRole("heading", { name: "参考资料" })).toBeInTheDocument();
    const sourceLink = screen.getByRole("link", { name: /慢性肾病指南/ });
    expect(sourceLink).toHaveAttribute("href", "https://example.test/ckd");
    expect(sourceLink).toHaveAttribute("target", "_blank");
    expect(sourceLink).toHaveAttribute("rel", "noreferrer noopener");
  });
});
