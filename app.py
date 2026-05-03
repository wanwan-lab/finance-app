# -*- coding: utf-8 -*-
"""
AIランウェイ診断 — 経営者向けランウェイ簡易試算（Streamlit）
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImprovementCuts:
    """改善シナリオ: 各項目の削減率（％）。適用後は 原価 × (1 − 率/100)。"""
    advertising_pct: float
    outsourcing_pct: float
    other_fixed_pct: float  # 「その他」「システム費」に適用


@dataclass(frozen=True)
class CostBreakdown:
    personnel: float  # 人件費
    rent: float  # 家賃
    advertising: float  # 広告費
    outsourcing: float  # 外注費
    system: float  # システム費
    other: float  # その他
    loan_repayment: float  # 借入返済
    tax_social: float  # 税金社保

    def total(self) -> float:
        return (
            self.personnel
            + self.rent
            + self.advertising
            + self.outsourcing
            + self.system
            + self.other
            + self.loan_repayment
            + self.tax_social
        )

    def apply_scenario(
        self,
        *,
        cost_multiplier: float = 1.0,
        improvement_cuts: Optional[ImprovementCuts] = None,
    ) -> "CostBreakdown":
        """費用増は全項目に倍率。改善案は広告・外注・その他固定（その他＋システム）のみ削減。"""
        if improvement_cuts is None:
            return CostBreakdown(
                personnel=self.personnel * cost_multiplier,
                rent=self.rent * cost_multiplier,
                advertising=self.advertising * cost_multiplier,
                outsourcing=self.outsourcing * cost_multiplier,
                system=self.system * cost_multiplier,
                other=self.other * cost_multiplier,
                loan_repayment=self.loan_repayment * cost_multiplier,
                tax_social=self.tax_social * cost_multiplier,
            )
        fa = 1.0 - improvement_cuts.advertising_pct / 100.0
        fo = 1.0 - improvement_cuts.outsourcing_pct / 100.0
        fx = 1.0 - improvement_cuts.other_fixed_pct / 100.0
        return CostBreakdown(
            personnel=self.personnel,
            rent=self.rent,
            advertising=max(0.0, self.advertising * fa),
            outsourcing=max(0.0, self.outsourcing * fo),
            system=max(0.0, self.system * fx),
            other=max(0.0, self.other * fx),
            loan_repayment=self.loan_repayment,
            tax_social=self.tax_social,
        )


@dataclass(frozen=True)
class RunwayResult:
    gross_burn: float
    net_burn: float
    runway_months: Optional[float]  # None = 資金減少なし
    runway_label: str
    shortfall_month: Optional[int]  # 初めて月末現金が0未満になる月（1始まり）、なければ None


def gross_profit(monthly_revenue: float, gross_margin_pct: float) -> float:
    return monthly_revenue * (gross_margin_pct / 100.0)


def compute_burns(monthly_revenue: float, gross_margin_pct: float, costs: CostBreakdown) -> tuple[float, float]:
    """Gross Burn = 月次支出合計。Net Burn = max(0, 支出 - 粗利)（認識ベース・サイトなしの当月）。"""
    expenditure = costs.total()
    gp = gross_profit(monthly_revenue, gross_margin_pct)
    net_raw = expenditure - gp
    net_burn = max(0.0, net_raw)
    return expenditure, net_burn


def revenue_in_month(base_monthly_sales: float, growth_pct: float, month_index: int) -> float:
    """month_index: 1〜12。当月売上を月1とする。"""
    g = growth_pct / 100.0
    return base_monthly_sales * ((1.0 + g) ** (month_index - 1))


def revenue_for_cash_month(
    calendar_month: int,
    collection_lag_months: int,
    base_monthly_sales: float,
    growth_pct: float,
) -> float:
    """
    入金サイト分ずらした「当月のキャッシュインに対応する売上」。
    認識月 index = calendar_month - lag。index < 1 のときは過去実績がないため入力の月次売上を使用。
    """
    accrual_month = calendar_month - collection_lag_months
    if accrual_month < 1:
        return base_monthly_sales
    return revenue_in_month(base_monthly_sales, growth_pct, accrual_month)


def expense_for_cash_month(
    calendar_month: int,
    payment_lag_months: int,
    monthly_expense_series: list[float],
) -> float:
    """
    支払サイト分ずらした「当月のキャッシュアウトに対応する支出」。
    monthly_expense_series[m-1] = 認識ベースの m 月目の月次支出合計。
    index < 1 のときはシミュレーション開始前とみなし、月1と同じ水準（入力ベース）を使用。
    """
    accrual_month = calendar_month - payment_lag_months
    if accrual_month < 1:
        accrual_month = 1
    return monthly_expense_series[accrual_month - 1]


def build_monthly_expense_series(costs: CostBreakdown, months: int) -> list[float]:
    """認識ベースの各月の固定費合計（シナリオ内では月ごとに一定）。"""
    e = costs.total()
    return [e for _ in range(months)]


def project_cash_monthly(
    initial_cash: float,
    base_monthly_sales: float,
    growth_pct: float,
    gross_margin_pct: float,
    costs: CostBreakdown,
    collection_lag_months: int,
    payment_lag_months: int,
    months: int = 12,
) -> pd.DataFrame:
    """
    月末現金残高の推移（サイト考慮）。
    粗利の入金は入金サイト分遅れ、固定費の支払は支払サイト分遅れた認識月の金額を当月キャッシュに反映。
    """
    expense_series = build_monthly_expense_series(costs, months)
    rows: list[dict] = []
    cash_start = initial_cash
    for m in range(1, months + 1):
        rev = revenue_in_month(base_monthly_sales, growth_pct, m)
        gp = gross_profit(rev, gross_margin_pct)
        exp = costs.total()

        rev_cash = revenue_for_cash_month(m, collection_lag_months, base_monthly_sales, growth_pct)
        gp_cash = gross_profit(rev_cash, gross_margin_pct)
        exp_cash = expense_for_cash_month(m, payment_lag_months, expense_series)

        net_burn = max(0.0, exp_cash - gp_cash)

        cash_end = cash_start - net_burn
        rows.append(
            {
                "月": m,
                "月次売上（認識）": rev,
                "粗利（認識）": gp,
                "入金サイト反映売上": rev_cash,
                "入金サイト反映粗利": gp_cash,
                "月次支出合計（認識）": exp,
                "支払サイト反映支出": exp_cash,
                "ネットバーン": net_burn,
                "月初現金": cash_start,
                "月末現金": cash_end,
            }
        )
        cash_start = cash_end
    return pd.DataFrame(rows)


def first_shortfall_month(df: pd.DataFrame) -> Optional[int]:
    neg = df[df["月末現金"] < 0]
    if neg.empty:
        return None
    return int(neg.iloc[0]["月"])


def runway_snapshot(
    initial_cash: float,
    monthly_revenue: float,
    growth_pct: float,
    gross_margin_pct: float,
    costs: CostBreakdown,
    collection_lag_months: int,
    payment_lag_months: int,
) -> RunwayResult:
    """月1のサイト調整後ネットバーンでランウェイを算出。"""
    expenditure = costs.total()
    expense_series = build_monthly_expense_series(costs, 12)
    rev_cash_m1 = revenue_for_cash_month(1, collection_lag_months, monthly_revenue, growth_pct)
    gp_cash_m1 = gross_profit(rev_cash_m1, gross_margin_pct)
    exp_cash_m1 = expense_for_cash_month(1, payment_lag_months, expense_series)
    net_burn = max(0.0, exp_cash_m1 - gp_cash_m1)

    df = project_cash_monthly(
        initial_cash,
        monthly_revenue,
        growth_pct,
        gross_margin_pct,
        costs,
        collection_lag_months,
        payment_lag_months,
        12,
    )
    short_m = first_shortfall_month(df)
    if net_burn <= 0:
        return RunwayResult(
            gross_burn=expenditure,
            net_burn=0.0,
            runway_months=None,
            runway_label="資金減少なし",
            shortfall_month=short_m,
        )
    months = initial_cash / net_burn
    return RunwayResult(
        gross_burn=expenditure,
        net_burn=net_burn,
        runway_months=months,
        runway_label=f"{months:.1f} ヶ月",
        shortfall_month=short_m,
    )


def risk_level(runway: RunwayResult) -> tuple[str, str, str]:
    """
    危険度: 安全 / 注意 / 危険 / 緊急
    戻り値: (ラベル, 説明短文, CSS的な色ヒント: green/amber/orange/red)
    """
    if runway.net_burn <= 0:
        if runway.shortfall_month is not None:
            return (
                "注意",
                "いまの月のペースでは現金は減りませんが、将来の試算でマイナス月が出ています。",
                "amber",
            )
        return "安全", "月の粗利で支出をカバーできています。", "green"
    m = runway.runway_months
    assert m is not None
    if m >= 12:
        return "安全", "現金はおおむね1年分以上の余裕があります。", "green"
    if m >= 6:
        return "注意", "半年以上は持ちますが、早めの見直しが安心です。", "amber"
    if m >= 3:
        return "危険", "数ヶ月単位で資金が減ります。優先順位をつけて対策を。", "orange"
    return "緊急", "すぐに資金が底をつく可能性があります。", "red"


def diagnostic_comment(base: RunwayResult, df_base: pd.DataFrame) -> tuple[str, str, str]:
    """現在の状態・主なリスク・優先アクション（ルールベース）。"""
    state_parts: list[str] = []
    risk_parts: list[str] = []
    action_parts: list[str] = []

    if base.net_burn <= 0:
        state_parts.append("いまのペースでは現金が減る心配はほぼありません（試算上）。")
        risk_parts.append("売上の急落や粗利率の悪化があると状況は変わります。")
        action_parts.append("それでも売上の分散と固定費の見える化は続けましょう。")
    else:
        state_parts.append(
            f"毎月おおよそ {base.net_burn:,.0f} 円ずつ現金が減る試算です（ネットバーン）。"
        )
        if base.runway_months is not None:
            state_parts.append(f"単純割りではランウェイは約 {base.runway_months:.1f} ヶ月です。")
        if base.shortfall_month is not None:
            state_parts.append(
                f"「{base.shortfall_month} ヶ月目」に初めて月末現金がマイナスになる試算です。"
            )
        else:
            state_parts.append("12ヶ月以内の試算では現金がマイナスにはなりません。")

        if base.gross_burn > 0 and base.net_burn / base.gross_burn > 0.7:
            risk_parts.append("支出のうち粗利で埋まらない割合が大きく、バーンが重い状態です。")
        if base.shortfall_month is not None and base.shortfall_month <= 6:
            risk_parts.append("半年以内に資金ショートの試算が出ています。")
        if not risk_parts:
            risk_parts.append("売上成長が止まるとランウェイは短くなりやすい点に注意です。")

        action_parts.append("まずは支出の大きい項目から見直し、入金タイミングも含めて確認してください。")
        if base.shortfall_month is not None:
            action_parts.append("借入・投資・売上改善のどれで穴を埋めるか、選択肢を並べるのが有効です。")

    return "\n".join(state_parts), "\n".join(risk_parts), "\n".join(action_parts)


def color_for_risk(key: str) -> str:
    return {"green": "#16a34a", "amber": "#ca8a04", "orange": "#ea580c", "red": "#dc2626"}.get(key, "#64748b")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AIランウェイ診断", layout="wide", initial_sidebar_state="expanded")

st.title("AIランウェイ診断")
st.caption("約3分で「あと何ヶ月もつか」をざっくり把握するためのダッシュボードです。")

st.info(
    "**入金サイト**を1ヶ月に設定すると、売上に紐づく粗利の入金がシミュレーション上で1ヶ月後ろにずれます"
    "（例: 2ヶ月目の入金は、認識ベースでは1ヶ月目の売上を元に計算）。"
    " **支払サイト**を1ヶ月にすると、固定費の支払が1ヶ月遅れた月の発生額として扱われます。"
    " シミュレーション開始前の月が参照される場合は、**入力の月次売上**および**月1の支出水準**で代用します。"
)

ScenarioKey = Literal["base", "downside", "cost_up", "improve"]


@dataclass(frozen=True)
class ScenarioSpec:
    key: ScenarioKey
    title: str
    revenue_factor: float
    cost_mult: float
    improvement_cuts: Optional[ImprovementCuts]


def scenario_costs_from_spec(base: CostBreakdown, spec: ScenarioSpec) -> CostBreakdown:
    return base.apply_scenario(
        cost_multiplier=spec.cost_mult,
        improvement_cuts=spec.improvement_cuts,
    )


def build_scenario_specs(
    downside_pct: float,
    cost_up_pct: float,
    improve_cuts: ImprovementCuts,
) -> dict[ScenarioKey, ScenarioSpec]:
    rf_down = 1.0 - downside_pct / 100.0
    cm_up = 1.0 + cost_up_pct / 100.0
    return {
        "base": ScenarioSpec("base", "Base Case（入力どおり）", 1.0, 1.0, None),
        "downside": ScenarioSpec(
            "downside",
            f"下振れ（売上 −{downside_pct:.0f}%）",
            rf_down,
            1.0,
            None,
        ),
        "cost_up": ScenarioSpec(
            "cost_up",
            f"費用増（支出 ＋{cost_up_pct:.0f}%）",
            1.0,
            cm_up,
            None,
        ),
        "improve": ScenarioSpec(
            "improve",
            "改善案（設定した削減率で広告・外注・その他固定費を圧縮）",
            1.0,
            1.0,
            improve_cuts,
        ),
    }


with st.sidebar:
    st.header("入力")
    cash = st.number_input("現預金残高（円）", min_value=0.0, value=8_000_000.0, step=100_000.0, format="%.0f")
    sales = st.number_input("月次売上（円／月）", min_value=0.0, value=3_000_000.0, step=50_000.0, format="%.0f")
    growth = st.number_input("売上成長率（％／月）", min_value=-50.0, value=0.0, step=0.5, format="%.1f")
    gm = st.number_input("粗利率（％）", min_value=0.0, max_value=100.0, value=40.0, step=1.0, format="%.1f")

    st.subheader("キャッシュのタイミング（サイト）")
    collection_lag = st.slider(
        "入金サイト（売上を計上してから現金が入るまでの月数）",
        min_value=0,
        max_value=3,
        value=0,
        help="0で当月入金。1なら粗利の入金が1ヶ月遅れます。",
    )
    payment_lag = st.slider(
        "支払サイト（費用を発生させてから現金が出ていくまでの月数）",
        min_value=0,
        max_value=3,
        value=0,
        help="0で当月支払。固定費の支払タイミングをずらして計算します。",
    )

    st.subheader("固定費の内訳（円／月）")
    c_personnel = st.number_input("人件費", min_value=0.0, value=1_200_000.0, step=10_000.0, format="%.0f")
    c_rent = st.number_input("家賃", min_value=0.0, value=300_000.0, step=10_000.0, format="%.0f")
    c_ad = st.number_input("広告費", min_value=0.0, value=150_000.0, step=10_000.0, format="%.0f")
    c_out = st.number_input("外注費", min_value=0.0, value=200_000.0, step=10_000.0, format="%.0f")
    c_sys = st.number_input("システム費", min_value=0.0, value=80_000.0, step=5_000.0, format="%.0f")
    c_other = st.number_input("その他", min_value=0.0, value=120_000.0, step=10_000.0, format="%.0f")
    c_loan = st.number_input("借入返済", min_value=0.0, value=100_000.0, step=10_000.0, format="%.0f")
    c_tax = st.number_input("税金・社保", min_value=0.0, value=250_000.0, step=10_000.0, format="%.0f")

    st.subheader("シミュレーション設定")
    downside_pct = st.slider("下振れシナリオの売上減少率（％）", 0.0, 80.0, 20.0, 1.0)
    cost_up_pct = st.slider("費用増シナリオの支出増加率（％）", 0.0, 80.0, 20.0, 1.0)
    st.caption("改善シナリオの削減率（％）")
    cut_ad = st.slider("広告費の削減率", 0.0, 90.0, 45.0, 1.0)
    cut_out = st.slider("外注費の削減率", 0.0, 90.0, 45.0, 1.0)
    cut_other_fix = st.slider("その他固定費（その他・システム費）の削減率", 0.0, 90.0, 12.0, 1.0)

base_costs = CostBreakdown(
    personnel=c_personnel,
    rent=c_rent,
    advertising=c_ad,
    outsourcing=c_out,
    system=c_sys,
    other=c_other,
    loan_repayment=c_loan,
    tax_social=c_tax,
)

improve_cuts = ImprovementCuts(
    advertising_pct=cut_ad,
    outsourcing_pct=cut_out,
    other_fixed_pct=cut_other_fix,
)
SCENARIO_SPECS = build_scenario_specs(downside_pct, cost_up_pct, improve_cuts)


def build_runway_for_scenario(
    key: ScenarioKey,
    collection_lag_m: int,
    payment_lag_m: int,
) -> tuple[RunwayResult, pd.DataFrame]:
    spec = SCENARIO_SPECS[key]
    rf = spec.revenue_factor
    costs = scenario_costs_from_spec(base_costs, spec)
    eff_sales = sales * rf
    r = runway_snapshot(
        cash,
        eff_sales,
        growth,
        gm,
        costs,
        collection_lag_m,
        payment_lag_m,
    )
    df = project_cash_monthly(
        cash,
        eff_sales,
        growth,
        gm,
        costs,
        collection_lag_m,
        payment_lag_m,
        12,
    )
    return r, df


base_result, df_base = build_runway_for_scenario("base", collection_lag, payment_lag)
label_risk, sub_risk, color_key = risk_level(base_result)
state_txt, risk_txt, action_txt = diagnostic_comment(base_result, df_base)

# --- ダッシュボード KPI ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("月の支出の合計（グロスバーン）", f"{base_result.gross_burn:,.0f} 円")
with kpi2:
    st.metric("月の現金の減り（ネットバーン）", f"{base_result.net_burn:,.0f} 円")
with kpi3:
    st.metric("ランウェイ（目安）", base_result.runway_label)
with kpi4:
    short_txt = "なし" if base_result.shortfall_month is None else f"{base_result.shortfall_month} ヶ月目"
    st.metric("資金ショートが出る月（試算）", short_txt)

st.markdown(
    f"""
<div style="padding:14px 18px;border-radius:10px;border-left:6px solid {color_for_risk(color_key)};
background:#0f172a08;margin:12px 0;">
<b>危険度：{label_risk}</b><br/><span style="opacity:.9">{sub_risk}</span>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("感応度分析（4シナリオ）")
scenario_rows: list[dict] = []
scenario_dfs: dict[str, pd.DataFrame] = {}

for sk, spec in SCENARIO_SPECS.items():
    rr, dff = build_runway_for_scenario(sk, collection_lag, payment_lag)
    scenario_dfs[spec.title] = dff
    scenario_rows.append(
        {
            "シナリオ": spec.title,
            "グロスバーン（円／月）": rr.gross_burn,
            "ネットバーン（円／月）": rr.net_burn,
            "ランウェイ": rr.runway_label,
            "資金ショート月（初回）": "—" if rr.shortfall_month is None else f"{rr.shortfall_month} ヶ月目",
        }
    )

df_compare = pd.DataFrame(scenario_rows)
df_compare_show = df_compare.copy()
for _col in ("グロスバーン（円／月）", "ネットバーン（円／月）"):
    df_compare_show[_col] = df_compare_show[_col].map(lambda x: f"{x:,.0f}")
st.dataframe(df_compare_show, use_container_width=True, hide_index=True)

# グラフ: 全シナリオの月末現金（サイト・シミュレーション設定を反映）
colors = ("#2563eb", "#dc2626", "#ca8a04", "#16a34a")
fig = go.Figure()
for i, (title, dff) in enumerate(scenario_dfs.items()):
    fig.add_trace(
        go.Scatter(
            x=dff["月"],
            y=dff["月末現金"],
            mode="lines+markers",
            name=title,
            line=dict(width=2, color=colors[i % len(colors)]),
        )
    )
fig.add_hline(
    y=0,
    line_color="red",
    line_width=2,
    line_dash="solid",
    annotation_text="0円ライン",
    annotation_position="bottom right",
)
fig.update_layout(
    title="月末現金残高の推移（4シナリオ・サイト反映）",
    xaxis_title="経過月（今から何ヶ月後か）",
    yaxis_title="円",
    hovermode="x unified",
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("診断コメント（ルールベース）")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**現在の状態**")
    st.write(state_txt)
with c2:
    st.markdown("**主なリスク**")
    st.write(risk_txt)
with c3:
    st.markdown("**優先アクション**")
    st.write(action_txt)

st.divider()
st.subheader("CSVダウンロード")

buf_forecast = io.StringIO()
df_base.to_csv(buf_forecast, index=False, encoding="utf-8-sig")
st.download_button(
    label="12ヶ月予測表（Base Case）をCSVでダウンロード",
    data=buf_forecast.getvalue().encode("utf-8-sig"),
    file_name="runway_12m_forecast_base.csv",
    mime="text/csv",
)

buf_scenarios = io.StringIO()
df_compare.to_csv(buf_scenarios, index=False, encoding="utf-8-sig")
st.download_button(
    label="シナリオ比較表をCSVでダウンロード",
    data=buf_scenarios.getvalue().encode("utf-8-sig"),
    file_name="runway_scenario_compare.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown(
    '<p style="font-size:0.85rem;color:#64748b;line-height:1.6;">'
    "本サービスは簡易試算であり、実際の資金繰り・税務・契約条件とは一致しない場合があります。"
    "重要な判断の前には、税理士・金融機関・経営コンサルタントなど専門家への相談を推奨します。"
    "</p>",
    unsafe_allow_html=True,
)
