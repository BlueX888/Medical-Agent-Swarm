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
});
