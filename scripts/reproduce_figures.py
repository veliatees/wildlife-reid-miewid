"""Reproduce summary charts (fig1, fig2, and fig5 selective abstention) from locked results JSONs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"


def make_figures() -> None:
    val_sum = json.loads((RES / "val_summary.json").read_text(encoding="utf-8"))
    test_report = json.loads((RES / "test_summary.json").read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: Imaging factors comparison
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=160)
    labels = [
        "Random split\n(val pool)",
        "Unseen days\n(test)",
        "Same cam\n(test)",
        "Cross cam\n(test)",
        "Nyala\nsame side",
        "Nyala\nopposite side",
    ]
    miew_vals = [
        val_sum["miewid_msv3"]["random_split"]["r1"],
        test_report["miewid_msv3"]["cross_day_test"]["r1"],
        test_report["miewid_msv3"]["same_cam_test"]["r1"],
        test_report["miewid_msv3"]["cross_cam_test"]["r1"],
        test_report["miewid_msv3"]["nyala_same_side_test"]["r1"],
        test_report["miewid_msv3"]["nyala_opposite_side_test"]["r1"],
    ]
    adapt_vals = [
        np.nan,
        test_report["miewid_adapted"]["cross_day_test"]["r1"],
        test_report["miewid_adapted"]["same_cam_test"]["r1"],
        test_report["miewid_adapted"]["cross_cam_test"]["r1"],
        np.nan,
        np.nan,
    ]
    mega_vals = [
        val_sum["megadescriptor_l384"]["random_split"]["r1"],
        test_report["megadescriptor_l384"]["cross_day_test"]["r1"],
        test_report["megadescriptor_l384"]["same_cam_test"]["r1"],
        test_report["megadescriptor_l384"]["cross_cam_test"]["r1"],
        test_report["megadescriptor_l384"]["nyala_same_side_test"]["r1"],
        test_report["megadescriptor_l384"]["nyala_opposite_side_test"]["r1"],
    ]
    x = np.arange(len(labels))
    w = 0.26
    b1 = ax.bar(x - w, [v * 100 for v in miew_vals], w, label="MiewID-msv3 (zero-shot)", color="#17365D")
    b2 = ax.bar(
        x,
        [v * 100 if not np.isnan(v) else 0 for v in adapt_vals],
        w,
        label="MiewID + Gallery Adapter (cows)",
        color="#2E7D32",
    )
    b3 = ax.bar(x + w, [v * 100 for v in mega_vals], w, label="MegaDescriptor-L-384*", color="#5B9BD5")
    for bars, vals in [(b1, miew_vals), (b2, adapt_vals), (b3, mega_vals)]:
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8)
    ax.set_ylim(0, 114)
    ax.set_ylabel("Rank-1 Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Effect of Time, Camera, and Viewpoint Factors on Bovid Re-ID")
    ax.legend(frameon=False, loc="upper right", fontsize=8.8)
    fig.text(
        0.01,
        0.005,
        "*MegaDescriptor-L-384 pre-training overlaps NyalaData (2023) but not MultiCamCows2024.",
        fontsize=7.8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(RES / "fig1_imaging_factors.png", dpi=160)
    plt.close(fig)

    # Fig 2: Field degradation impact
    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=160)
    dlabels = ["Clean", "Low light\n(0.35x)", "Blur\n(sigma=3)", "Tight crop\n(70%)"]
    modes = ["clean", "low_light", "blur", "crop70"]
    pretty = {
        "miewid_msv3": "MiewID-msv3 (zero-shot)",
        "miewid_adapted": "MiewID + Gallery Adapter",
        "megadescriptor_l384": "MegaDescriptor-L-384",
    }
    offsets = {
        "megadescriptor_l384": [(0.0, 4.2), (0.0, 4.2), (0.0, 4.2), (0.0, -5.8)],
        "miewid_msv3": [(-0.20, -1.2), (-0.21, -0.5), (0.0, -5.8), (-0.16, 3.8)],
        "miewid_adapted": [(0.0, -6.2), (0.0, -6.2), (0.0, 4.2), (0.19, 1.0)],
    }
    for nm, col, mk in [
        ("miewid_msv3", "#17365D", "o"),
        ("miewid_adapted", "#2E7D32", "s"),
        ("megadescriptor_l384", "#5B9BD5", "^"),
    ]:
        ys = [test_report[nm]["cross_day_test"]["r1"] * 100] + [
            test_report[nm]["degradations_test"][m]["r1"] * 100 for m in modes[1:]
        ]
        ax.plot(range(len(dlabels)), ys, marker=mk, lw=2.2, ms=7, label=pretty[nm], color=col)
        for xi, yv in enumerate(ys):
            dx, dy = offsets[nm][xi]
            ax.text(xi + dx, yv + dy, f"{yv:.1f}%", ha="center", va="center", fontsize=8.4, fontweight="medium", color=col)
    ax.set_xticks(range(len(dlabels)))
    ax.set_xticklabels(dlabels)
    ax.set_xlim(-0.42, 3.42)
    ax.set_ylim(0, 106)
    ax.set_ylabel("Rank-1 Accuracy (%) — Locked Test Days")
    ax.set_title("Sensitivity to Handheld Field Degradations (MultiCamCows2024 Test)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(RES / "fig2_degradations.png", dpi=160)
    plt.close(fig)

    # Fig 5: Selective prediction (risk-coverage via locked id_margin)
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=160)
    cov_labels = ["100% coverage\n(no abstention)", "90% coverage\n(flag lowest 10%)", "80% coverage\n(flag lowest 20%)"]
    sel_offsets = {
        "megadescriptor_l384": [(0.0, 1.1), (0.0, 1.1), (0.0, 1.1)],
        "miewid_msv3": [(-0.15, 0.3), (-0.16, 0.6), (-0.17, 0.0)],
        "miewid_adapted": [(0.0, -1.4), (0.16, -0.9), (0.0, -1.4)],
    }
    for nm, col, mk in [
        ("miewid_msv3", "#17365D", "o"),
        ("miewid_adapted", "#2E7D32", "s"),
        ("megadescriptor_l384", "#5B9BD5", "^"),
    ]:
        ys = [
            test_report[nm]["cross_day_test"]["r1"] * 100,
            test_report[nm]["selective_r1_at_90pct_cov"] * 100,
            test_report[nm]["selective_r1_at_80pct_cov"] * 100,
        ]
        ax.plot(range(3), ys, marker=mk, lw=2.2, ms=7, label=pretty[nm], color=col)
        for xi, yv in enumerate(ys):
            dx, dy = sel_offsets[nm][xi]
            ax.text(xi + dx, yv + dy, f"{yv:.1f}%", ha="center", va="center", fontsize=8.6, fontweight="medium", color=col)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cov_labels)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(82, 101)
    ax.set_ylabel("Closed-Set Rank-1 Accuracy (%)")
    ax.set_title("Selective Prediction Using Locked Identity Margin (MultiCamCows2024 Test)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "fig5_selective_abstention.png", dpi=160)
    plt.close(fig)
    print("Reproduced fig1, fig2, and fig5 cleanly.")


if __name__ == "__main__":
    make_figures()
