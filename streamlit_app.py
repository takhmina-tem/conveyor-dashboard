# streamlit_app.py
import os
from datetime import datetime, date, time as dtime, timedelta, timezone
from typing import Optional
from io import BytesIO
import zipfile

import streamlit as st
import pandas as pd
import altair as alt
from zoneinfo import ZoneInfo

# ===== Настройки времени =====
TZ = ZoneInfo("Asia/Aqtobe")  # для внутренних конверсий, в UI не показываем

# «Целевая» рамка на вчера: 4 сентября 14:00–19:00 (локально)
TARGET_DAY_LOCAL   = date(2025, 9, 4)
TARGET_START_HOUR  = 14
TARGET_END_HOUR    = 19   # правая граница (5 часов окно)

# «Живое» окно: последние N минут
LIVE_MINUTES = 60

# ===== Страница =====
st.set_page_config(page_title="Система отслеживания и учёта картофеля", page_icon="🥔", layout="wide")

# ===== Ключи Supabase =====
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

# ===== Стили =====
st.markdown("""
<style>
  .block-container { padding-top: 2.0rem; }
  .hdr { display:flex; justify-content:space-between; align-items:center; }
  .hdr h1 { margin:0; font-size:26px; font-weight:800; letter-spacing:.3px; }
  hr { margin: 10px 0 18px 0; opacity:.25; }
</style>
""", unsafe_allow_html=True)

def header():
    col_l, col_r = st.columns([3,1])
    with col_l:
        st.markdown("<div class='hdr'><h1>Система отслеживания и учёта картофеля</h1></div>", unsafe_allow_html=True)
    with col_r:
        if st.button("↻ Обновить страницу", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    st.markdown("<hr/>", unsafe_allow_html=True)

# ===== Время/конверсии =====
def to_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def local_range_to_utc(d: date, start_h: int, end_h: int) -> tuple[datetime, datetime]:
    start_local = datetime.combine(d, dtime(hour=start_h)).replace(tzinfo=TZ)
    end_local   = datetime.combine(d, dtime(hour=end_h)).replace(tzinfo=TZ)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

def live_window_utc(minutes: int = LIVE_MINUTES) -> tuple[datetime, datetime]:
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    return now_utc - timedelta(minutes=minutes), now_utc

def remap_to_target_window(df: pd.DataFrame,
                           live_start_utc: datetime,
                           target_start_local: datetime) -> pd.DataFrame:
    """Переносит ts (UTC-aware) в 'вчера 14:00 + оффсет от начала live-окна'."""
    if df.empty or "ts" not in df.columns:
        return df
    tgt_start_utc = target_start_local.astimezone(timezone.utc)
    # оффсет от начала live-окна в секундах
    offs = (df["ts"] - live_start_utc).dt.total_seconds().clip(lower=0)
    df = df.copy()
    df["ts_disp"] = tgt_start_utc + pd.to_timedelta(offs, unit="s")
    return df

# ===== Чтение данных =====
@st.cache_data(ttl=5)
def fetch_events(point: Optional[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Возвращает ts (UTC-aware), point, potato_id, width/height (см)."""
    if not USE_SUPABASE:
        return pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm","width","height"])
    try:
        q = _sb.table("events").select("*").order("ts", desc=False)
        if point:
            q = q.eq("point", point)
        rows = q.gte("ts", to_aware_utc(start_dt).isoformat())\
                .lte("ts", to_aware_utc(end_dt).isoformat())\
                .execute().data
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        for c in ("width_cm","height_cm","width","height"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Ошибка чтения из Supabase: {e}")
        return pd.DataFrame(columns=["ts","point","potato_id","width_cm","height_cm","width","height"])

# ===== Подготовка размеров (мм) =====
def add_mm_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # приоритет: *_cm; иначе width/height считаем в см и переводим в мм
    if "width_cm" in d.columns:
        d["width_mm"] = d["width_cm"] * 10.0
    elif "width" in d.columns:
        d["width_mm"] = d["width"] * 10.0
    else:
        d["width_mm"] = pd.NA

    if "height_cm" in d.columns:
        d["height_mm"] = d["height_cm"] * 10.0
    elif "height" in d.columns:
        d["height_mm"] = d["height"] * 10.0
    else:
        d["height_mm"] = pd.NA

    d["width_mm"]  = pd.to_numeric(d["width_mm"], errors="coerce")
    d["height_mm"] = pd.to_numeric(d["height_mm"], errors="coerce")
    return d

# ===== Агрегации =====
def hour_counts_collected(df_disp: pd.DataFrame) -> pd.DataFrame:
    """Группировка по ЧАСАМ по колонке ts_disp (уже перемапленной в 'вчера')."""
    if df_disp.empty or "ts_disp" not in df_disp.columns:
        return pd.DataFrame({"hour": [], "collected": []})
    ts_local = df_disp["ts_disp"].dt.tz_convert(TZ)  # из UTC в локальное, чтобы красиво на оси
    hours_naive = ts_local.dt.floor("h").dt.tz_localize(None)
    g = (
        pd.DataFrame({"hour": hours_naive, "potato_id": df_disp["potato_id"]})
        .groupby("hour", as_index=False)
        .agg(collected=("potato_id", "nunique"))
    )
    return g

# ===== Категории (мм) =====
CAT_LABELS = ["<30 мм", "30–40 мм", "40–50 мм", "50–60 мм", ">60 мм"]
CAT_BINS_MM = [0, 30, 40, 50, 60, 1_000_000]

def bins_table_mm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS)})
    d = add_mm_columns(df)
    if "width_mm" not in d.columns or d["width_mm"].dropna().empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS)})
    cut = pd.cut(d["width_mm"].fillna(-1), bins=CAT_BINS_MM, labels=CAT_LABELS, right=False, include_lowest=True)
    vc = cut.value_counts().reindex(CAT_LABELS).fillna(0).astype(int)
    return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [int(vc[c]) for c in CAT_LABELS]})

# ===== Визуализации =====
def render_hour_chart(df_disp: pd.DataFrame):
    hc = hour_counts_collected(df_disp)
    if hc.empty:
        st.info("Нет данных в текущем (перемапленном) окне.")
        return pd.DataFrame()
    chart = (
        alt.Chart(hc)
        .mark_bar()
        .encode(
            x=alt.X("hour:T", title="Дата и час (отображается как вчера)"),
            y=alt.Y("collected:Q", title="Собрано (шт)"),
            tooltip=[alt.Tooltip("hour:T", title="Час"), alt.Tooltip("collected:Q", title="Шт")],
        )
        .properties(height=320, padding={"top": 10, "right": 12, "bottom": 44, "left": 8})
        .configure_axis(labelFontSize=12, titleFontSize=12)
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    return hc

# ===== Экспорт =====
def make_excel_bytes(hour_df: pd.DataFrame, bins_df: pd.DataFrame) -> tuple[bytes, str, str]:
    try:
        import xlsxwriter  # noqa
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as w:
            hour_df.to_excel(w, index=False, sheet_name="Поток по часам (вчера)")
            bins_df.to_excel(w, index=False, sheet_name="Категории (мм)")
        return buf.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        pass
    try:
        import openpyxl  # noqa
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm") as w:
            hour_df.to_excel(w, index=False, sheet_name="Поток по часам (вчера)")
            bins_df.to_excel(w, index=False, sheet_name="Категории (мм)")
        return buf.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        pass
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hour_flow.csv", hour_df.to_csv(index=False))
        zf.writestr("bins_mm.csv",  bins_df.to_csv(index=False))
    return buf.getvalue(), "zip", "application/zip"

# ===== Страница =====
def page_dashboard():
    header()

    # Текстовый маркер периода
    st.caption("Отображаем текущие поступления так, как будто они идут вчера, 04.09, с 14:00 до 19:00.")

    # 1) Живое окно (UTC)
    live_start_utc, live_end_utc = live_window_utc(LIVE_MINUTES)

    # 2) «Вчерашняя» рамка (локально -> UTC)
    target_start_local = datetime.combine(TARGET_DAY_LOCAL, dtime(hour=TARGET_START_HOUR)).replace(tzinfo=TZ)
    target_end_local   = datetime.combine(TARGET_DAY_LOCAL, dtime(hour=TARGET_END_HOUR)).replace(tzinfo=TZ)

    # 3) Читаем за последний час только точку A (это и есть «Собрано» по ТЗ)
    df_live = fetch_events("A", live_start_utc, live_end_utc)

    # 4) Переносим метки времени в «вчерашнюю» рамку
    df_disp = remap_to_target_window(df_live, live_start_utc, target_start_local)

    # 5) Метрика «Собрано (шт)»
    collected_total = df_disp["potato_id"].nunique() if not df_disp.empty else 0
    st.metric("Собрано (шт)", value=f"{collected_total}")

    # 6) Поток по часам (уже по ts_disp)
    st.markdown("### Поток по часам (отображение: вчера)")
    hc = render_hour_chart(df_disp)

    # 7) Категории (мм) — считаем из размеров
    st.markdown("### Таблица по категориям (мм)")
    bins_df = bins_table_mm(df_disp)
    st.dataframe(bins_df, use_container_width=True)

    # 8) Экспорт
    hour_export = (
        hc.rename(columns={"hour": "Дата и час", "collected": "Собрано (шт)"})
        if not hc.empty else pd.DataFrame(columns=["Дата и час","Собрано (шт)"])
    )
    file_bytes, ext, mime = make_excel_bytes(hour_export, bins_df)
    st.download_button(
        label="Скачать отчёт",
        data=file_bytes,
        file_name=f"potato_report_as_if_2025-09-04_14-19.{ext}",
        mime=mime,
        use_container_width=True
    )

# ===== MAIN =====
def main():
    page_dashboard()

if __name__ == "__main__":
    main()
