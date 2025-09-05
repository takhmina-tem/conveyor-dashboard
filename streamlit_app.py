# streamlit_app.py
import os
import time
import random
from datetime import datetime, date, time as dtime, timedelta, timezone
from typing import Optional
from io import BytesIO
import zipfile

import streamlit as st
import pandas as pd
import altair as alt

from zoneinfo import ZoneInfo
TZ = ZoneInfo("Asia/Aqtobe")  # GMT+5 (замени при необходимости)

# ====== Вспомогательные функции времени ======
def local_day_bounds_to_utc(d: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(d, dtime.min).replace(tzinfo=TZ)
    end_local   = datetime.combine(d, dtime.max).replace(tzinfo=TZ)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

def _ensure_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def week_bounds(d: date) -> tuple[date, date]:
    start = d - timedelta(days=d.weekday())
    end = start + timedelta(days=6)
    return start, end

# ====== Базовые настройки ======
st.set_page_config(
    page_title="Система отслеживания и учёта картофеля",
    page_icon="🥔",
    layout="wide",
)

# ====== Ключи (secrets/environment) ======
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", os.getenv("SUPABASE_ANON_KEY"))
DEFAULT_BATCH = st.secrets.get("DEFAULT_BATCH", os.getenv("DEFAULT_BATCH", ""))  # чтобы сразу фильтровался «текущий запуск»

USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_ANON_KEY)
_sb = None
if USE_SUPABASE:
    try:
        from supabase import create_client
        _sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        st.warning(f"Не удалось инициализировать Supabase: {e}")
        USE_SUPABASE = False

# ====== Оформление ======
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

# ====== Сессия/роутер (упрощённо) ======
if "authed" not in st.session_state:
    st.session_state["authed"] = True
if "route" not in st.session_state:
    st.session_state["route"] = "app"
if "day_picker" not in st.session_state:
    st.session_state["day_picker"] = date.today()

def go(page: str):
    st.session_state["route"] = page
    st.rerun()

# ====== Чтение данных (кэш с TTL для лёгкого пуллинга) ======
@st.cache_data(ttl=5)  # небольшой TTL для ручного обновления
def fetch_events(point: Optional[str], start_dt: datetime, end_dt: datetime, batch: Optional[str] = None) -> pd.DataFrame:
    """Возвращает ts (UTC-aware), point, potato_id, width_cm, height_cm, batch."""
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
        # стандарт: делаем AWARE-UTC
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        for col in ("width_cm","height_cm"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Ошибка чтения из Supabase: {e}")
        return pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm","batch"])

# ====== Агрегации (теперь ts всегда UTC-aware) ======
def hour_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame({"hour": [], "count": []})
    ts_local = df["ts"].dt.tz_convert(TZ)
    hours_naive = ts_local.dt.floor("h").dt.tz_localize(None)
    return (
        pd.DataFrame({"hour": hours_naive, "potato_id": df["potato_id"]})
          .groupby("hour", as_index=False)
          .agg(count=("potato_id", "nunique"))
    )

def day_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame({"day": [], "count": []})
    ts_local = df["ts"].dt.tz_convert(TZ)
    days_naive = ts_local.dt.floor("D").dt.tz_localize(None)
    return (
        pd.DataFrame({"day": days_naive, "potato_id": df["potato_id"]})
          .groupby("day", as_index=False)
          .agg(count=("potato_id", "nunique"))
    )

# ====== Категории ======
CATEGORIES = ["<30", "30–40", "40–50", "50–60", ">60"]

def bins_table(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    """A=Изначально, B=Собрано. Сейчас Потери не считаем, Собрано = Изначально."""
    def count_bins(df: pd.DataFrame) -> pd.Series:
        if df.empty or ("width_cm" not in df.columns):
            return pd.Series({c: 0 for c in CATEGORIES})
        bins = [0,30,40,50,60,10_000]
        labels = CATEGORIES
        cut = pd.cut(df["width_cm"].fillna(-1), bins=bins, labels=labels, right=False, include_lowest=True)
        vc = cut.value_counts().reindex(labels).fillna(0).astype(int)
        return vc

    A = count_bins(dfA)
    # B = A (по требованию «Собрано = Изначально»)
    B = A.copy()

    return pd.DataFrame({
        "Категория":  CATEGORIES,
        "Изначально": [int(A[c]) for c in CATEGORIES],
        "Собрано":    [int(B[c]) for c in CATEGORIES],
    })

# ====== Чарт по часам (Собрано = Изначально) ======
def render_hour_chart_grouped(dfA: pd.DataFrame):
    ha = hour_counts(dfA).rename(columns={"count": "initial"})
    if ha.empty:
        st.info("Нет данных за выбранный период.")
        return pd.DataFrame()

    # Собрано = Изначально
    merged = ha.copy()
    merged["collected"] = merged["initial"]

    long_df = pd.concat([
        merged[["hour", "collected"]].rename(columns={"collected": "Значение"}).assign(Сегмент="Собрано"),
        merged[["hour", "initial"]].rename(columns={"initial": "Значение"}).assign(Сегмент="Изначально"),
    ], ignore_index=True)

    x_axis = alt.X(
        "hour:T",
        title="Дата и час",
        axis=alt.Axis(titlePadding=24, labelOverlap=True, labelFlush=True, titleAnchor="start"),
    )

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=x_axis,
            y=alt.Y("Значение:Q", title="Количество", stack="zero"),
            color=alt.Color("Сегмент:N", title="", scale=alt.Scale(domain=["Собрано", "Изначально"])),
            tooltip=[alt.Tooltip("hour:T", title="Час"), alt.Tooltip("Сегмент:N"), alt.Tooltip("Значение:Q", title="Шт")],
        )
        .properties(height=320, padding={"top": 10, "right": 12, "bottom": 44, "left": 8})
        .configure_axis(labelFontSize=12, titleFontSize=12)
    )

    st.altair_chart(chart, use_container_width=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    return merged

# ====== Excel выгрузка ======
def make_excel_bytes(hour_df: pd.DataFrame, bins_df: pd.DataFrame) -> tuple[bytes, str, str]:
    try:
        import xlsxwriter  # noqa: F401
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as writer:
            hour_df.to_excel(writer, index=False, sheet_name="Поток по часам")
            bins_df.to_excel(writer, index=False, sheet_name="Категории")
        return buf.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        pass

    try:
        import openpyxl  # noqa: F401
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm") as writer:
            hour_df.to_excel(writer, index=False, sheet_name="Поток по часам")
            bins_df.to_excel(writer, index=False, sheet_name="Категории")
        return buf.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        pass

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Поток по часам.csv", hour_df.to_csv(index=False))
        zf.writestr("Категории.csv",    bins_df.to_csv(index=False))
    return buf.getvalue(), "zip", "application/zip"

# ====== Главная страница ======
def page_dashboard_online():
    header()

    # верхняя панель: batch и кнопка обновления
    left, mid, right = st.columns([1.5, 1, 1])
    with left:
        batch_tag = st.text_input("batch-тег (фильтр по текущему запуску)", value=DEFAULT_BATCH or "")
    with mid:
        if st.button("↻ Обновить страницу", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with right:
        if USE_SUPABASE:
            try:
                _sb.table("events").select("potato_id").limit(1).execute()
                st.caption("✅ Supabase: OK")
            except Exception as e:
                st.warning(f"⚠️ Supabase не отвечает: {e}")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ==== LIVE (последний час, только текущий batch) ====
    st.subheader("Live: последние 60 минут (текущий запуск)")
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    live_start = now_utc - timedelta(hours=1)

    dfA_live = fetch_events("A", live_start, now_utc, batch=batch_tag or None)
    # по требованиям: «Собрано = Изначально», B сейчас не используем

    live_init = dfA_live["potato_id"].nunique() if not dfA_live.empty else 0
    live_coll = live_init  # Собрано = Изначально

    m1, m2 = st.columns(2)
    m1.metric("Изначально (шт)", value=f"{live_init}")
    m2.metric("Собрано (шт)", value=f"{live_coll}")

    with st.expander("Последние события (A)"):
        if not dfA_live.empty:
            df_tail = dfA_live.sort_values("ts", ascending=False).head(30).copy()
            df_tail["ts"] = df_tail["ts"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
            df_view(df_tail[["ts","potato_id","width_cm","height_cm","batch"]])
        else:
            st.info("Нет событий за последний час для выбранного batch.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ==== ДЕНЬ (по выбранной локальной дате, фильтруем по batch) ====
    st.subheader("День (локальная дата, текущий запуск)")
    dcol1, dcol2 = st.columns([1, 1])
    with dcol1:
        st.date_input("Дата", key="day_picker")
    with dcol2:
        st.caption(f"Часовой пояс: {TZ}")

    day = st.session_state["day_picker"]
    start_day_utc, end_day_utc = local_day_bounds_to_utc(day)

    dfA = fetch_events("A", start_day_utc, end_day_utc, batch=batch_tag or None)

    # метрики за день
    total_initial = dfA["potato_id"].nunique() if not dfA.empty else 0
    total_collected = total_initial  # Собрано = Изначально

    dm1, dm2 = st.columns(2)
    dm1.metric("Изначально (шт)", value=f"{total_initial}")
    dm2.metric("Собрано (шт)", value=f"{total_collected}")

    # поток по часам (A; collected = initial)
    st.markdown("### Поток по часам")
    merged_hours = render_hour_chart_grouped(dfA)

    # Категории
    bins_df = bins_table(dfA, dfA)  # B=A
    st.markdown("### Таблица по категориям")
    df_view(bins_df[["Категория","Изначально","Собрано"]])

    # Excel-отчёт (часы + категории)
    ha = hour_counts(dfA).rename(columns={"count": "Изначально"})
    hour_export = ha.rename(columns={"hour": "Дата и час"}).copy()
    hour_export["Собрано"] = hour_export["Изначально"]

    file_bytes, ext, mime = make_excel_bytes(hour_export, bins_df)
    st.download_button(
        label="Скачать отчёт (Excel/CSV)",
        data=file_bytes,
        file_name=f"potato_report_{day.isoformat()}." + ext,
        mime=mime,
        use_container_width=True
    )

# ====== App ======
def page_app():
    page_dashboard_online()

# ====== Main ======
def main():
    page_app()

if __name__ == "__main__":
    main()
