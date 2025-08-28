# streamlit_app.py
import os
import random
from datetime import datetime, date, time, timedelta, timezone
from typing import Optional

import streamlit as st
import pandas as pd
import altair as alt

# ====== РЕЖИМ ДЕМО ДАННЫХ (НЕ ВЛИЯЕТ НА АВТОРИЗАЦИЮ) ======
FORCE_DEMO_DATA = True  # ← поставьте False, чтобы читать реальные данные из БД

# ====== БАЗОВЫЕ НАСТРОЙКИ ======
st.set_page_config(
    page_title="Система отслеживания и учёта картофеля",
    page_icon="🥔",
    layout="wide",
)

# ====== КЛЮЧИ (через .streamlit/secrets.toml или env) ======
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", os.getenv("SUPABASE_ANON_KEY"))

USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_ANON_KEY)
_sb = None
if USE_SUPABASE:
    try:
        from supabase import create_client
        _sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        st.warning(f"Не удалось инициализировать Supabase: {e}")
        USE_SUPABASE = False

# ====== ОФОРМЛЕНИЕ ======
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.25rem; }
      .hdr { display:flex; justify-content:space-between; align-items:center; }
      .hdr h1 { margin:0; font-size:26px; font-weight:800; letter-spacing:.3px; }
      .hdr .sub { opacity:.8 }
      .hdr + .spacer { height: 10px; }
      hr { margin: 10px 0 22px 0; opacity:.25; }
      .stDateInput, .stNumberInput, .stTextInput { margin-bottom: .35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def header(sub: str | None = None):
    st.markdown(
        f"<div class='hdr'><h1>Система отслеживания и учёта картофеля</h1>{f'<div class=\"sub\">{sub}</div>' if sub else ''}</div>",
        unsafe_allow_html=True
    )
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

def df_view(df: pd.DataFrame, caption: str = ""):
    if caption:
        st.caption(caption)
    st.dataframe(df, use_container_width=True)

# ====== СЕССИЯ/РОУТЕР ======
if "authed" not in st.session_state:
    st.session_state["authed"] = False
if "route" not in st.session_state:
    st.session_state["route"] = "login"  # 'login' | 'app'

def go(page: str):
    st.session_state["route"] = page
    st.rerun()

# ====== ДАТЫ ======
def _ensure_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

# ====== ЧТЕНИЕ ДАННЫХ (если USE_SUPABASE=True) ======
def fetch_events(point: Optional[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Возвращает ts (naive UTC), point, potato_id, width_cm, height_cm."""
    if not USE_SUPABASE:
        return pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm"])
    try:
        q = _sb.table("events").select("*").order("ts", desc=False)
        if point:
            q = q.eq("point", point)
        start_iso = _ensure_aware_utc(start_dt).isoformat()
        end_iso   = _ensure_aware_utc(end_dt).isoformat()
        data = q.gte("ts", start_iso).lte("ts", end_iso).execute().data
        df = pd.DataFrame(data) if data else pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm"])
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")\
                          .dt.tz_convert("UTC").dt.tz_localize(None)
        for col in ("width_cm","height_cm"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Ошибка чтения из Supabase: {e}")
        return pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm"])

def hour_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame({"hour":[], "count":[]})
    return (
        df.assign(hour=pd.to_datetime(df["ts"]).dt.floor("h"))
          .groupby("hour", as_index=False)
          .agg(count=("potato_id", "nunique"))
    )

# ====== КАТЕГОРИИ ======
CATEGORIES = ["<30", "30–40", "40–50", "50–60", ">60"]

def bins_table(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    """A=Изначально, B=Собрано. Возвращает таблицу по категориям."""
    def count_bins(df: pd.DataFrame) -> pd.Series:
        if df.empty or ("width_cm" not in df.columns):
            return pd.Series({c: 0 for c in CATEGORIES})
        bins = [0,30,40,50,60,10_000]
        labels = CATEGORIES
        cut = pd.cut(
            df["width_cm"].fillna(-1),
            bins=bins, labels=labels, right=False, include_lowest=True
        )
        vc = cut.value_counts().reindex(labels).fillna(0).astype(int)
        return vc

    A = count_bins(dfA)   # Изначально (A)
    B = count_bins(dfB)   # Собрано   (B)
    losses = (A - B).clip(lower=0)
    loss_pct = pd.Series({c: (0.0 if A[c]==0 else round(losses[c]/A[c]*100, 1)) for c in CATEGORIES})

    return pd.DataFrame({
        "Категория":    CATEGORIES,
        "Изначально":   [int(A[c])      for c in CATEGORIES],
        "Потери (шт)":  [int(losses[c]) for c in CATEGORIES],
        "Собрано":      [int(B[c])      for c in CATEGORIES],
        "% потери":     [float(loss_pct[c]) for c in CATEGORIES],
    })

# ====== ДЕМО-ДАННЫЕ ======
def demo_generate(day: date, base: int = 620, jitter: int = 90, seed: int = 42):
    """
    Реалистичный демо-поток:
      - A (вход) ~ N(base, jitter)
      - B (собрано) = A * U(0.70, 0.85)
      - width_cm ~ N(52..53, 7.5..8.5) с отсечением [25, 75]
    """
    rng = random.Random(seed + int(day.strftime("%Y%m%d")))
    hours = [datetime.combine(day, time(h,0)) for h in range(24)]
    rowsA, rowsB = [], []
    pid = 1
    for ts in hours:
        countA = max(0, int(rng.gauss(base, jitter)))
        countB = int(countA * rng.uniform(0.70, 0.85))
        for _ in range(countA):
            width = max(25.0, min(75.0, rng.gauss(52.0, 8.5)))
            rowsA.append({"ts": ts + timedelta(minutes=rng.randint(0,59)), "point":"A", "potato_id":pid, "width_cm":width, "height_cm":width*0.7})
            pid += 1
        for _ in range(countB):
            width = max(25.0, min(75.0, rng.gauss(53.0, 7.5)))
            rowsB.append({"ts": ts + timedelta(minutes=rng.randint(0,59)), "point":"B", "potato_id":pid, "width_cm":width, "height_cm":width*0.7})
            pid += 1
    return pd.DataFrame(rowsA), pd.DataFrame(rowsB)

# ====== ЛОГИН (без повторного header) ======
def page_login():
    st.subheader("Вход в систему")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("E-mail", placeholder="you@company.com")
        password = st.text_input("Пароль", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Войти")
    if submitted:
        ok = True
        if USE_SUPABASE:
            try:
                resp = _sb.auth.sign_in_with_password({"email": email, "password": password})
                ok = bool(getattr(resp, "user", None))
                if not ok:
                    st.error("Неверная почта или пароль.")
            except Exception as e:
                st.error(f"Ошибка авторизации: {e}")
                ok = False
        if ok:
            st.session_state["authed"] = True
            st.session_state["route"] = "app"
            st.rerun()
    st.caption("Доступ выдаётся администраторами.")

# ====== ЧАРТ ПО ЧАСАМ (STACKED: B внизу, (A-B) сверху; сумма = A) ======
def render_hour_chart_grouped(dfA: pd.DataFrame, dfB: pd.DataFrame):
    ha = hour_counts(dfA).rename(columns={"count": "initial"})   # A = Изначально
    hb = hour_counts(dfB).rename(columns={"count": "collected"}) # B = Итого (Б)
    merged = pd.merge(ha, hb, on="hour", how="outer").sort_values("hour")
    if merged.empty:
        st.info("Нет данных за выбранный период.")
        return

    merged[["initial", "collected"]] = merged[["initial", "collected"]].fillna(0).astype(int)
    merged["diff"] = (merged["initial"] - merged["collected"]).clip(lower=0)

    # Готовим long-формат для стекования
    long_df = pd.concat([
        merged[["hour", "collected"]].rename(columns={"collected": "Значение"}).assign(Сегмент="Итого (B)"),
        merged[["hour", "diff"]].rename(columns={"diff": "Значение"}).assign(Сегмент="Изначально (A)"),
    ], ignore_index=True)

    x_axis = alt.X(
        "hour:T",
        title="Дата и час",
        axis=alt.Axis(
            titlePadding=24,
            labelOverlap=True,
            labelFlush=True,
            titleAnchor="start"
        ),
    )

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=x_axis,
            y=alt.Y("Значение:Q", title="Количество", stack="zero"),
            color=alt.Color(
                "Сегмент:N",
                title="",
                scale=alt.Scale(domain=["Итого (B)", "Изначально (A)"])
            ),
            tooltip=[
                alt.Tooltip("hour:T", title="Час"),
                alt.Tooltip("Сегмент:N"),
                alt.Tooltip("Значение:Q", title="В сегменте"),
            ],
        )
        .properties(
            height=320,
            padding={"top": 10, "right": 12, "bottom": 44, "left": 8}
        )
        .configure_axis(
            labelFontSize=12,
            titleFontSize=12,
        )
    )

    st.altair_chart(chart, use_container_width=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ====== КАЛЬКУЛЯТОР КАПИТАЛА (текстовый ввод, плейсхолдер "0") ======
DEFAULT_WEIGHT_G = {"<30": 0.0, "30–40": 0.0, "40–50": 0.0, "50–60": 0.0, ">60": 0.0}
DEFAULT_PRICE_KG = {"<30": 0.0, "30–40": 0.0, "40–50": 0.0, "50–60": 0.0, ">60": 0.0}

def _parse_float(s: str) -> float:
    if s is None:
        return 0.0
    s = s.strip().replace(",", ".")
    if s == "":
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

def capital_calculator(bins_df: pd.DataFrame):
    st.markdown("### Калькулятор капитала")

    counts = dict(zip(bins_df["Категория"], bins_df["Собрано"]))

    col_w = st.columns(5)
    col_p = st.columns(5)
    weights_g = {}
    prices_kg = {}

    # Текстовый ввод: виден «серый» 0 (placeholder), но поле пустое — можно просто печатать
    for i, cat in enumerate(CATEGORIES):
        with col_w[i]:
            raw_w = st.text_input(
                f"Вес ({cat}), г/шт",
                value=st.session_state.get(f"calc_w_{cat}", ""),
                placeholder="0",
                key=f"calc_w_{cat}",
            )
            weights_g[cat] = _parse_float(raw_w)

        with col_p[i]:
            raw_p = st.text_input(
                f"Цена ({cat}), тг/кг",
                value=st.session_state.get(f"calc_p_{cat}", ""),
                placeholder="0",
                key=f"calc_p_{cat}",
            )
            prices_kg[cat] = _parse_float(raw_p)

    kg_totals = {cat: (counts.get(cat, 0) * weights_g.get(cat, 0.0)) / 1000.0 for cat in CATEGORIES}
    subtotals = {cat: kg_totals[cat] * prices_kg.get(cat, 0.0) for cat in CATEGORIES}
    total_sum = round(sum(subtotals.values()), 2)

    calc_df = pd.DataFrame({
        "Категория":        CATEGORIES,
        "Собрано (шт)":     [int(counts.get(c, 0)) for c in CATEGORIES],
        "Вес, г/шт":        [weights_g[c] for c in CATEGORIES],
        "Итого, кг":        [round(kg_totals[c], 3) for c in CATEGORIES],
        "Цена, тг/кг":      [prices_kg[c] for c in CATEGORIES],
        "Сумма, тг":        [round(subtotals[c], 2) for c in CATEGORIES],
    })
    df_view(calc_df)
    st.subheader(f"Итого капитал: **{total_sum:,.2f} тг**".replace(",", " "))

# ====== ГЛАВНАЯ СТРАНИЦА ======
def page_dashboard_online():
    # убрали подзаголовок "Онлайн-данные"
    header()

    c_top1, c_top2, c_top3 = st.columns([1.3,1,1])
    with c_top1:
        day = st.date_input("Дата", value=date.today())
    with c_top2:
        st.empty()
    with c_top3:
        if st.button("Выйти"):
            st.session_state["authed"] = False
            go("login")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    start = datetime.combine(day, time.min).replace(tzinfo=timezone.utc)
    end   = datetime.combine(day, time.max).replace(tzinfo=timezone.utc)

    if FORCE_DEMO_DATA:
        dfA, dfB = demo_generate(day)
    else:
        dfA = fetch_events("A", start, end)
        dfB = fetch_events("B", start, end)
        if dfA.empty and dfB.empty:
            dfA, dfB = demo_generate(day)

    total_initial = dfA["potato_id"].nunique() if not dfA.empty else 0
    total_collected = dfB["potato_id"].nunique() if not dfB.empty else 0
    total_losses = max(0, total_initial - total_collected)

    m1, m2, m3 = st.columns(3)
    m1.metric("Изначально (шт)", value=f"{total_initial}")
    m2.metric("Потери (шт)", value=f"{total_losses}")
    m3.metric("Собрано (шт)", value=f"{total_collected}")

    st.markdown("### Поток по часам")
    render_hour_chart_grouped(dfA, dfB)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("### Таблица по количеству")
    bins_df = bins_table(dfA, dfB)
    df_view(bins_df[["Категория","Изначально","Потери (шт)","Собрано","% потери"]])

    capital_calculator(bins_df)

# ====== APP (без вкладки видео-демо) ======
def page_app():
    page_dashboard_online()

# ====== MAIN ======
def main():
    # Если нет ключей Supabase — открываем приложение без авторизации
    if not USE_SUPABASE:
        st.session_state["authed"] = True
        st.session_state["route"] = "app"

    route = st.session_state.get("route", "login")
    authed = st.session_state.get("authed", False)

    if route == "login" and USE_SUPABASE and not authed:
        header("Авторизация")
        page_login()
        return

    page_app()

if __name__ == "__main__":
    main()
