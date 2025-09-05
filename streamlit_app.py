# streamlit_app.py
import os
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
    "<style>.block-container{padding-top:1.2rem}.hdr{display:flex;justify-content:space-between;align-items:center}.hdr h1{margin:0;font-size:26px;font-weight:800;letter-spacing:.3px}.hdr .sub{opacity:.8}hr{margin:8px 0 16px 0;opacity:.25}div[aria-live='polite']{display:none!important}</style>",
    unsafe_allow_html=True,
)

def header(sub=""):
    st.markdown(
        '<div class="hdr"><h1>Система отслеживания и учёта картофеля</h1>'
        f'<div class="sub">{sub}</div></div><hr/>',
        unsafe_allow_html=True
    )

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

def safe_datetime_input(label: str, value: datetime) -> datetime:
    if hasattr(st, "datetime_input"):
        dt = st.datetime_input(label, value=value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    # Фоллбек для старых Streamlit
    c1, c2 = st.columns([1.2, 1])
    with c1:
        d = st.date_input(label + " — дата", value=value.date())
    with c2:
        t = st.time_input(label + " — время", value=value.time())
    return datetime.combine(d, t).replace(tzinfo=timezone.utc)

# ====== ЧТЕНИЕ ДАННЫХ ======
def fetch_events(point: Optional[str], start_dt: datetime, end_dt: datetime, batch: Optional[str] = None) -> pd.DataFrame:
    """Возвращает ts (naive UTC), point, potato_id, width_cm, height_cm, batch."""
    if not USE_SUPABASE:
        return pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm","batch"])

    try:
        q = _sb.table("events").select("*").order("ts", desc=False)
        if point:
            q = q.eq("point", point)
        if batch:
            q = q.eq("batch", batch)

        start_iso = _ensure_aware_utc(start_dt).isoformat()
        end_iso   = _ensure_aware_utc(end_dt).isoformat()
        data = q.gte("ts", start_iso).lte("ts", end_iso).execute().data

        df = pd.DataFrame(data) if data else pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm","batch"])
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")\
                          .dt.tz_convert("UTC").dt.tz_localize(None)
        for col in ("width_cm","height_cm"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Ошибка чтения из Supabase: {e}")
        return pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm","batch"])

def hour_bars(df: pd.DataFrame):
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame({"hour":[], "count":[]})
    return (df.assign(hour=pd.to_datetime(df["ts"]).dt.floor("h"))
              .groupby("hour", as_index=False)
              .agg(count=("potato_id", "nunique")))

def bins_table(dfA: pd.DataFrame, dfB: pd.DataFrame):
    cols = [">60","50–60","40–50","30–40"]

    def count_bins(df: pd.DataFrame):
        if df.empty or ("width_cm" not in df.columns) or df["width_cm"].isna().all():
            n = df["potato_id"].nunique() if "potato_id" in df.columns else len(df)
            q = n // 4; r = n - 3*q
            return pd.Series({">60":r, "50–60":q, "40–50":q, "30–40":q})
        bins = [0,30,40,50,60,10_000]
        labels_all = ["<30","30–40","40–50","50–60",">60"]
        cut = pd.cut(df["width_cm"].fillna(-1), bins=bins, labels=labels_all, right=False, include_lowest=True)
        vc = cut.value_counts().reindex(labels_all).fillna(0).astype(int)
        return pd.Series({
            ">60":   int(vc[">60"]),
            "50–60": int(vc["50–60"]),
            "40–50": int(vc["40–50"]),
            "30–40": int(vc["30–40"]),
        })

    A = count_bins(dfA)
    B = count_bins(dfB)
    total = A
    collected = B
    discarded = (A - B).clip(lower=0)
    loss_pct = pd.Series({c: (0.0 if total[c]==0 else round(discarded[c]/total[c]*100, 1)) for c in cols})

    return pd.DataFrame({
        "Категория":  cols,
        "Общее":      [int(total[c])     for c in cols],
        "Собрано":    [int(collected[c]) for c in cols],
        "Выброшено":  [int(discarded[c]) for c in cols],
        "% Потери":   [float(loss_pct[c]) for c in cols],
    })

# ====== СТРАНИЦЫ ======
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
    st.caption("Доступ выдаётся администраторами. Обратитесь к ответственному сотруднику.")

def render_hour_chart(df):
    bars = hour_bars(df)
    if bars.empty:
        st.info("Нет данных за выбранный период.")
        return
    chart = (
        alt.Chart(bars)
        .mark_bar()
        .encode(
            x=alt.X("hour:T", title="Дата и час"),
            y=alt.Y("count:Q", title="Количество"),
            tooltip=["hour:T","count:Q"]
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

def page_dashboard_online():
    header("Онлайн-данные")

    c1, c2, c3, c4 = st.columns([1,1.2,1.2,2])
    with c1:
        point = st.selectbox("Точка", ["A","B"], index=0)
    with c2:
        day = st.date_input("Дата", value=date.today())
    with c3:
        st.caption("Фильтрация по batch помогает смотреть конкретный прогон")
    with c4:
        if st.button("Выйти"):
            st.session_state["authed"] = False
            go("login")

    # фильтр по batch (необязательный)
    batch_tag = st.text_input("batch-тег (необязательно)", value="")

    # загрузка данных (ТОЛЬКО реальные данные, без демо)
    start = datetime.combine(day, time.min).replace(tzinfo=timezone.utc)
    end   = datetime.combine(day, time.max).replace(tzinfo=timezone.utc)

    dfA = fetch_events("A", start, end, batch=batch_tag or None)
    dfB = fetch_events("B", start, end, batch=batch_tag or None)

    if dfA.empty and dfB.empty:
        st.info("Нет данных за выбранную дату и фильтр batch.")
        st.stop()

    # Баннер состояния
    st.caption(f"Источник: {SUPABASE_URL} · batch: {batch_tag or '—'} · период (UTC): {start} — {end}")

    # График по выбранной точке
    df_current = dfA if point == "A" else dfB
    st.markdown("### Поток по часам")
    render_hour_chart(df_current)

    # Таблица по категориям (A как общее, B как собрано)
    st.markdown("### Таблица по количеству")
    df_view(bins_table(dfA, dfB))

# ====== MAIN (без вкладки «Видео-демо») ======
def main():
    if not USE_SUPABASE:
        header("Настройка")
        st.error("SUPABASE_URL / SUPABASE_ANON_KEY не заданы. Укажи их в Secrets или переменных окружения.")
        st.stop()

    route = st.session_state.get("route", "login")
    authed = st.session_state.get("authed", False)

    if route == "login" and not authed:
        page_login()
        return

    page_dashboard_online()

if __name__ == "__main__":
    main()
