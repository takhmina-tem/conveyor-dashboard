# streamlit_app.py
import os
import random
from datetime import datetime, date, time, timedelta, timezone
from typing import Optional

import streamlit as st
import pandas as pd
import altair as alt

# ====== БАЗОВЫЕ НАСТРОЙКИ ======
st.set_page_config(
    page_title="Система отслеживания и учёта картофеля",
    page_icon="🥔",
    layout="wide",
)

# ====== КЛЮЧИ (через Streamlit Secrets или env) ======
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

# ====== ВИДЕО-ПУТИ (Google Drive) ======
A_RAW  = "https://drive.google.com/uc?export=download&id=17_mf6Nn-BbRIC0FHl3fT5lWtTU8CUlEV"
B_RAW  = "https://drive.google.com/uc?export=download&id=1pJQoeqci4r3CnVRywGZWvHARFZ5JBwo1"
A_ANNO = "https://drive.google.com/uc?export=download&id=1HZ9U806VOdBeoiiAR_gF0ojabeZPCaWI"
B_ANNO = "https://drive.google.com/uc?export=download&id=1nI-4HNaXodkW9xnznikVvBwyFdITv2Yp"

# ====== ОФОРМЛЕНИЕ (улучшена видимость вкладок) ======
st.markdown(
    """
    <style>
      .block-container { padding-top: .6rem; }
      .hdr { display:flex; justify-content:space-between; align-items:center }
      .hdr h1 { margin:0; font-size:26px; font-weight:800; letter-spacing:.3px }
      hr { margin:8px 0 16px 0; opacity:.25 }
      /* не скрываем табы */
      .stTabs { margin-top: .2rem !important; }
      .stTabs [data-baseweb="tab-list"] { overflow-x: auto; }
      /* скрыть служебные live-элементы спиннеров */
      div[aria-live='polite']{ display:none!important }
    </style>
    """,
    unsafe_allow_html=True,
)

def header(sub: str | None = None):
    st.markdown(
        '<div class="hdr"><h1>Система отслеживания и учёта картофеля</h1></div><hr/>',
        unsafe_allow_html=True
    )
    if sub:
        st.caption(sub)

def df_view(df: pd.DataFrame, caption: str = ""):
    if caption:
        st.caption(caption)
    st.dataframe(df, use_container_width=True)

def video_player(url: str):
    st.markdown(
        f"""
        <video width="100%" controls>
          <source src="{url}" type="video/mp4">
          Ваш браузер не поддерживает видео.
        </video>
        """,
        unsafe_allow_html=True,
    )

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

# ====== ЧТЕНИЕ ДАННЫХ ======
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

# ====== ТАБЛИЦА КАТЕГОРИЙ ======
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

    A = count_bins(dfA)  # Изначально (точка A)
    B = count_bins(dfB)  # Собрано   (точка B)
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
def demo_generate(day: date, base: int = 600, jitter: int = 120, seed: int = 42):
    rng = random.Random(seed + int(day.strftime("%Y%m%d")))
    hours = [datetime.combine(day, time(h,0)) for h in range(24)]
    rowsA, rowsB = [], []
    pid = 1
    for ts in hours:
        countA = max(0, int(rng.gauss(base, jitter)))
        countB = int(countA * rng.uniform(0.6, 0.85))
        for _ in range(countA):
            width = max(25.0, min(75.0, rng.gauss(52.0, 9.5)))
            rowsA.append({"ts": ts + timedelta(minutes=rng.randint(0,59)), "point":"A", "potato_id":pid, "width_cm":width, "height_cm":width*0.7})
            pid += 1
        for _ in range(countB):
            width = max(25.0, min(75.0, rng.gauss(53.0, 8.5)))
            rowsB.append({"ts": ts + timedelta(minutes=rng.randint(0,59)), "point":"B", "potato_id":pid, "width_cm":width, "height_cm":width*0.7})
            pid += 1
    return pd.DataFrame(rowsA), pd.DataFrame(rowsB)

# ====== ЛОГИН ======
def page_login():
    header("Авторизация")
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
            go("app")
    st.caption("Доступ выдаётся администраторами.")

# ====== ЧАРТ ПО ЧАСАМ ======
def render_hour_chart_grouped(dfA: pd.DataFrame, dfB: pd.DataFrame):
    """Единый бар-чарт: Изначально (A) vs Собрано (B) по часам."""
    ha = hour_counts(dfA).rename(columns={"count": "initial"})   # A = Изначально
    hb = hour_counts(dfB).rename(columns={"count": "collected"}) # B = Собрано
    merged = pd.merge(ha, hb, on="hour", how="outer").sort_values("hour")
    if merged.empty:
        st.info("Нет данных за выбранный период.")
        return
    merged[["initial", "collected"]] = merged[["initial", "collected"]].fillna(0).astype(int)
    long_df = merged.melt(
        id_vars="hour",
        value_vars=["initial", "collected"],
        var_name="kind",
        value_name="value"
    )
    kind_map = {"initial": "Изначально", "collected": "Собрано"}
    long_df["Тип"] = long_df["kind"].map(kind_map)
    long_df = long_df.drop(columns=["kind"]).rename(columns={"value": "Значение"})
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("hour:T", title="Дата и час"),
            y=alt.Y("Значение:Q", title="Количество"),
            color=alt.Color("Тип:N", title=""),
            tooltip=[
                alt.Tooltip("hour:T", title="Час"),
                alt.Tooltip("Тип:N"),
                alt.Tooltip("Значение:Q")
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

# ====== КАЛЬКУЛЯТОР КАПИТАЛА (в тг) ======
def capital_calculator(bins_df: pd.DataFrame):
    st.markdown("### Калькулятор капитала")
    st.caption("Задайте цену за единицу для каждой категории (тг/шт). Подсчёт ведётся по столбцу «Собрано».")
    counts = dict(zip(bins_df["Категория"], bins_df["Собрано"]))
    cols = st.columns(5)
    default_prices = {"<30": 0.0, "30–40": 0.2, "40–50": 0.25, "50–60": 0.3, ">60": 0.35}
    prices = {}
    for i, cat in enumerate(CATEGORIES):
        with cols[i]:
            prices[cat] = st.number_input(f"Цена ({cat}), тг/шт", min_value=0.0, step=0.01,
                                          value=default_prices.get(cat, 0.0), key=f"price_{cat}")

    subtotals = {cat: float(counts.get(cat, 0)) * float(prices.get(cat, 0.0)) for cat in CATEGORIES}
    total = round(sum(subtotals.values()), 2)

    calc_df = pd.DataFrame({
        "Категория":      CATEGORIES,
        "Собрано (шт)":   [int(counts.get(c, 0)) for c in CATEGORIES],
        "Цена, тг/шт":    [prices[c] for c in CATEGORIES],
        "Сумма, тг":      [round(subtotals[c], 2) for c in CATEGORIES],
    })
    df_view(calc_df)
    st.subheader(f"Итого капитал: **{total:,.2f} тг**".replace(",", " "))

# ====== ГЛАВНАЯ ВКЛАДКА ======
def page_dashboard_online():
    header()

    # ---- Верхние метрики: Изначально / Потери / Собрано ----
    c_top1, c_top2, c_top3 = st.columns([1.2,1,1])
    with c_top1:
        day = st.date_input("Дата", value=date.today())
    with c_top2:
        pass
    with c_top3:
        if st.button("Выйти"):
            st.session_state["authed"] = False
            go("login")

    # Источник данных:
    start = datetime.combine(day, time.min).replace(tzinfo=timezone.utc)
    end   = datetime.combine(day, time.max).replace(tzinfo=timezone.utc)

    dfA = fetch_events("A", start, end) if USE_SUPABASE else pd.DataFrame()
    dfB = fetch_events("B", start, end) if USE_SUPABASE else pd.DataFrame()
    if (dfA.empty and dfB.empty):
        dfA, dfB = demo_generate(day)

    # ---- Метрики в самом верху (по запросу): Изначально / Потери / Собрано ----
    total_initial = dfA["potato_id"].nunique() if not dfA.empty else 0
    total_collected = dfB["potato_id"].nunique() if not dfB.empty else 0
    total_losses = max(0, total_initial - total_collected)

    m1, m2, m3 = st.columns(3)
    m1.metric("Изначально (шт)", value=f"{total_initial}")
    m2.metric("Потери (шт)", value=f"{total_losses}")
    m3.metric("Собрано (шт)", value=f"{total_collected}")

    # ---- Поток по часам (общий чарт с 2 цветами) ----
    st.markdown("### Поток по часам")
    render_hour_chart_grouped(dfA, dfB)

    # ---- Таблица по количеству (переупорядоченные колонки) ----
    st.markdown("### Таблица по количеству")
    bins_df = bins_table(dfA, dfB)
    df_view(bins_df[["Категория","Изначально","Потери (шт)","Собрано","% потери"]])

    # ---- Калькулятор капитала (тг) ----
    capital_calculator(bins_df)

def page_demo_from_videos():
    header("Видео-демо (A/B)")

    st.markdown("#### Ролики")
    left, right = st.columns(2)
    with left:
        st.caption("Точка A — исходник")
        video_player(A_RAW)
        st.caption("Точка A — аннотированное")
        video_player(A_ANNO)
    with right:
        st.caption("Точка B — исходник")
        video_player(B_RAW)
        st.caption("Точка B — аннотированное")
        video_player(B_ANNO)

    st.divider()

    st.markdown("#### Данные по роликам (если есть события в БД)")
    start_dt = datetime.now(timezone.utc) - timedelta(hours=2)
    end_dt   = datetime.now(timezone.utc)
    dfA = fetch_events("A", start_dt, end_dt)
    dfB = fetch_events("B", start_dt, end_dt)

    if dfA.empty and dfB.empty:
        st.info("Данных за интервал нет или БД не подключена.")
        return

    st.markdown("##### Поток по часам")
    render_hour_chart_grouped(dfA, dfB)

    st.markdown("##### Таблица по количеству")
    df_view(bins_table(dfA, dfB)[["Категория","Изначально","Потери (шт)","Собрано","% потери"]])

# ====== ОБЩАЯ СТРАНИЦА-ПРИЛОЖЕНИЕ ======
def page_app():
    # табы наверху экрана, хедер внутри вкладок — чтобы сами вкладки всегда были видны
    tab1, tab2 = st.tabs(["Онлайн-данные", "Видео-демо (A/B)"])
    with tab1:
        page_dashboard_online()
    with tab2:
        page_demo_from_videos()

# ====== MAIN ======
def main():
    # Если нет ключей Supabase — сразу открываем приложение без авторизации
    if not USE_SUPABASE:
        st.session_state["authed"] = True
        st.session_state["route"] = "app"

    route = st.session_state.get("route", "login")
    authed = st.session_state.get("authed", False)

    if route == "login" and USE_SUPABASE and not authed:
        page_login()
        return

    page_app()

if __name__ == "__main__":
    main()
