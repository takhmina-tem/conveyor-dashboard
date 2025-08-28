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

# ====== ВИДЕО-ПУТИ (Google Drive) ======
A_RAW  = "https://drive.google.com/uc?export=download&id=17_mf6Nn-BbRIC0FHl3fT5lWtTU8CUlEV"
B_RAW  = "https://drive.google.com/uc?export=download&id=1pJQoeqci4r3CnVRywGZWvHARFZ5JBwo1"
A_ANNO = "https://drive.google.com/uc?export=download&id=1HZ9U806VOdBeoiiAR_gF0ojabeZPCaWI"
B_ANNO = "https://drive.google.com/uc?export=download&id=1nI-4HNaXodkW9xnznikVvBwyFdITv2Yp"

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

def video_player(url: str):
    # HTML-видеоплеер надёжнее работает с прямыми ссылками GDrive
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
            width = max(28.0, min(75.0, rng.gauss(52.0, 9.5)))
            rowsA.append({"ts": ts + timedelta(minutes=rng.randint(0,59)), "point":"A", "potato_id":pid, "width_cm":width, "height_cm":width*0.7})
            pid += 1
        for _ in range(countB):
            width = max(28.0, min(75.0, rng.gauss(53.0, 8.5)))
            rowsB.append({"ts": ts + timedelta(minutes=rng.randint(0,59)), "point":"B", "potato_id":pid, "width_cm":width, "height_cm":width*0.7})
            pid += 1
    return pd.DataFrame(rowsA), pd.DataFrame(rowsB)

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
    st.caption(
        "Доступ выдаётся администраторами. "
        + ("(Supabase подключён)" if USE_SUPABASE else "(Supabase ключи не заданы — авторизация отключена)")
    )

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
        demo_mode = st.toggle("Демо-режим", value=True, help="Если БД пуста — сгенерировать красивую картину.")
    with c4:
        if st.button("Выйти"):
            st.session_state["authed"] = False
            go("login")

    # (Опционально) фильтр по batch на онлайновой вкладке
    batch_tag = st.text_input("batch-тег (необязательно)", value="")

    if demo_mode:
        dfA, dfB = demo_generate(day)
    else:
        start = datetime.combine(day, time.min).replace(tzinfo=timezone.utc)
        end   = datetime.combine(day, time.max).replace(tzinfo=timezone.utc)
        dfA = fetch_events("A", start, end, batch=batch_tag or None)
        dfB = fetch_events("B", start, end, batch=batch_tag or None)
        if dfA.empty and dfB.empty:
            dfA, dfB = demo_generate(day)

    df_current = dfA if point == "A" else dfB

    st.markdown("### Поток по часам")
    render_hour_chart(df_current)

    st.markdown("### Таблица по количеству")
    df_view(bins_table(dfA, dfB))

def page_demo_from_videos():
    header("Видео-демо (A/B)")

    # --- Ролики ---
    st.markdown("#### Ролики")
    left, right = st.columns(2)
    with left:
        st.caption("Точка A — исходник")
        video_player(A_RAW)    # без os.path.exists
        st.caption("Точка A — аннотированное")
        video_player(A_ANNO)
    with right:
        st.caption("Точка B — исходник")
        video_player(B_RAW)
        st.caption("Точка B — аннотированное")
        video_player(B_ANNO)

    st.divider()

    # --- Данные по роликам (только время и точка; batch не используется) ---
    st.markdown("#### Данные по роликам")
    info = st.container()
    c1, c2, c3 = st.columns([1.1, 1.1, 1.1])
    with c1:
        point_for_view = st.selectbox("Точка для графика", ["A","B"], index=0)
    with c2:
        start_dt = safe_datetime_input("Начало интервала (UTC)", value=datetime.now(timezone.utc) - timedelta(hours=2))
    with c3:
        end_dt = safe_datetime_input("Конец интервала (UTC)", value=datetime.now(timezone.utc))

    dfA = fetch_events("A", start_dt, end_dt, batch=None)
    dfB = fetch_events("B", start_dt, end_dt, batch=None)

    if dfA.empty and dfB.empty:
        with info:
            st.info("Данных за заданный интервал пока нет. Проверь интервал времени (UTC) и запуски A/B.")
        if st.button("Обновить"):
            st.rerun()
        st.markdown("### Поток по часам"); st.info("Нет данных.")
        st.markdown("### Таблица по количеству"); st.info("Нет данных.")
        st.stop()

    df_current = dfA if point_for_view == "A" else dfB

    st.markdown("### Поток по часам")
    render_hour_chart(df_current)

    st.markdown("### Таблица по количеству")
    df_view(bins_table(dfA, dfB))

    with st.expander("ДЕТАЛИ за час"):
        hour_sel = st.time_input("Час (локальное начало)", value=time(10,0))
        chosen_day = start_dt.date()
        hstart = datetime.combine(chosen_day, hour_sel)
        hend   = hstart + timedelta(hours=1) - timedelta(milliseconds=1)
        maskA = (dfA["ts"].between(hstart, hend)) if not dfA.empty else pd.Series([], dtype=bool)
        maskB = (dfB["ts"].between(hstart, hend)) if not dfB.empty else pd.Series([], dtype=bool)
        df_view(
            bins_table(
                dfA[maskA] if not dfA.empty else dfA,
                dfB[maskB] if not dfB.empty else dfB
            ),
            "Только за выбранный час"
        )

# ====== ОБЩАЯ СТРАНИЦА-ПРИЛОЖЕНИЕ ======
def page_app():
    sub = "Supabase: ✅ авторизация включена" if USE_SUPABASE else "Supabase: ⛔ ключи не заданы (авторизация отключена)"
    header(sub)
    tab1, tab2 = st.tabs(["Онлайн-данные", "Видео-демо (A/B)"])
    with tab1:
        page_dashboard_online()
    with tab2:
        page_demo_from_videos()

# ====== MAIN ======
def main():
    # Если нет ключей Supabase — открываем сразу приложение без авторизации
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
