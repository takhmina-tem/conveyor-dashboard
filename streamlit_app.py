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
TZ = ZoneInfo("Asia/Aqtobe")  # используется только для внутренней конвертации времени

# ====== Период по умолчанию (вчера, 4 сентября 14:00–19:00, локально) ======
DEFAULT_LOCAL_DATE = date(2025, 9, 4)
DEFAULT_START_HOUR = 14
DEFAULT_END_HOUR   = 19  # не включительно последняя минута, но мы берём до 19:00:00 max

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

# ====== СТИЛИ ======
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.25rem; }
      .hdr { display:flex; justify-content:space-between; align-items:center; }
      .hdr h1 { margin:0; font-size:26px; font-weight:800; letter-spacing:.3px; }
      .hdr .right { display:flex; gap:.5rem; }
      .hdr + .spacer { height: 10px; }
      hr { margin: 10px 0 22px 0; opacity:.25; }
      .stDateInput, .stNumberInput, .stTextInput { margin-bottom: .35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def header():
    col_l, col_r = st.columns([3,1])
    with col_l:
        st.markdown(
            "<div class='hdr'><h1>Система отслеживания и учёта картофеля</h1></div>",
            unsafe_allow_html=True
        )
    with col_r:
        if st.button("↻ Обновить страницу", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    st.markdown("<hr/>", unsafe_allow_html=True)

# ====== ВРЕМЯ/КОНВЕРСИИ ======
def to_aware_utc(dt: datetime) -> datetime:
    """Гарантированно UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def local_range_to_utc(d: date, start_h: int, end_h: int) -> tuple[datetime, datetime]:
    start_local = datetime.combine(d, dtime(hour=start_h, minute=0, second=0)).replace(tzinfo=TZ)
    end_local   = datetime.combine(d, dtime(hour=end_h,   minute=0, second=0)).replace(tzinfo=TZ)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

# ====== ЧТЕНИЕ ДАННЫХ ======
@st.cache_data(ttl=5)
def fetch_events(point: Optional[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """ts (UTC-aware), point, potato_id, width/height (в БД могут быть width_cm/height_cm ИЛИ width/height)."""
    if not USE_SUPABASE:
        return pd.DataFrame(columns=["ts","point","potato_id","width","height","width_cm","height_cm"])
    try:
        q = _sb.table("events").select("*").order("ts", desc=False)
        if point:
            q = q.eq("point", point)
        data = q.gte("ts", to_aware_utc(start_dt).isoformat())\
                .lte("ts", to_aware_utc(end_dt).isoformat())\
                .execute().data
        df = pd.DataFrame(data) if data else pd.DataFrame()
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")  # AWARE UTC
        # Числовые поля
        for col in ("width_cm","height_cm","width","height"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Ошибка чтения из Supabase: {e}")
        return pd.DataFrame(columns=["ts","point","potato_id","width","height","width_cm","height_cm"])

# ====== АГРЕГАЦИИ ======
def hour_counts_collected(df: pd.DataFrame) -> pd.DataFrame:
    """Группировка по часам для 'Собрано (шт)'. Берём уникальные potato_id за час."""
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame({"hour": [], "collected": []})
    ts_local = df["ts"].dt.tz_convert(TZ)
    hours_naive = ts_local.dt.floor("h").dt.tz_localize(None)
    g = (
        pd.DataFrame({"hour": hours_naive, "potato_id": df["potato_id"]})
        .groupby("hour", as_index=False)
        .agg(collected=("potato_id", "nunique"))
    )
    return g

# ====== КАТЕГОРИИ (в миллиметрах) ======
# Диапазоны в МИЛЛИМЕТРАХ:
CAT_LABELS = ["<30 мм", "30–40 мм", "40–50 мм", "50–60 мм", ">60 мм"]
CAT_BINS_MM = [0, 30, 40, 50, 60, 1_000_000]  # правый открытый

def add_mm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт width_mm/height_mm из возможных колонок (width_cm/height_cm или width/height)."""
    df = df.copy()
    # width_cm/height_cm приоритетно; иначе используем width/height. Всё считаем как см -> мм (×10).
    if "width_mm" not in df.columns:
        if "width_cm" in df.columns:
            df["width_mm"] = df["width_cm"] * 10.0
        elif "width" in df.columns:
            df["width_mm"] = df["width"] * 10.0
        else:
            df["width_mm"] = pd.NA
    if "height_mm" not in df.columns:
        if "height_cm" in df.columns:
            df["height_mm"] = df["height_cm"] * 10.0
        elif "height" in df.columns:
            df["height_mm"] = df["height"] * 10.0
        else:
            df["height_mm"] = pd.NA
    # к числам
    df["width_mm"]  = pd.to_numeric(df["width_mm"], errors="coerce")
    df["height_mm"] = pd.to_numeric(df["height_mm"], errors="coerce")
    return df

def bins_table_mm_collected(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица по категориям (мм). Используем ширину (width_mm)."""
    if df.empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS)})
    d = add_mm_columns(df)
    if "width_mm" not in d.columns or d["width_mm"].dropna().empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS)})

    cut = pd.cut(
        d["width_mm"].fillna(-1),
        bins=CAT_BINS_MM, labels=CAT_LABELS, right=False, include_lowest=True
    )
    vc = cut.value_counts().reindex(CAT_LABELS).fillna(0).astype(int)
    return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [int(vc[c]) for c in CAT_LABELS]})

# ====== ЧАРТ (Собрано) ======
def render_hour_chart_collected(df: pd.DataFrame):
    hc = hour_counts_collected(df)
    if hc.empty:
        st.info("Нет данных за выбранный период.")
        return pd.DataFrame()

    chart = (
        alt.Chart(hc)
        .mark_bar()
        .encode(
            x=alt.X("hour:T", title="Дата и час"),
            y=alt.Y("collected:Q", title="Собрано (шт)"),
            tooltip=[alt.Tooltip("hour:T", title="Час"), alt.Tooltip("collected:Q", title="Шт")],
        )
        .properties(height=320, padding={"top": 10, "right": 12, "bottom": 44, "left": 8})
        .configure_axis(labelFontSize=12, titleFontSize=12)
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    return hc

# ====== Excel выгрузка ======
def make_excel_bytes(hour_df: pd.DataFrame, bins_df: pd.DataFrame) -> tuple[bytes, str, str]:
    try:
        import xlsxwriter  # noqa: F401
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as writer:
            hour_df.to_excel(writer, index=False, sheet_name="Поток по часам")
            bins_df.to_excel(writer, index=False, sheet_name="Категории (мм)")
        return buf.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        pass

    try:
        import openpyxl  # noqa: F401
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm") as writer:
            hour_df.to_excel(writer, index=False, sheet_name="Поток по часам")
            bins_df.to_excel(writer, index=False, sheet_name="Категории (мм)")
        return buf.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        pass

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Поток по часам.csv", hour_df.to_csv(index=False))
        zf.writestr("Категории_мм.csv",  bins_df.to_csv(index=False))
    return buf.getvalue(), "zip", "application/zip"

# ====== ГЛАВНАЯ СТРАНИЦА ======
def page_dashboard():
    header()

    # Период (жёстко задан по ТЗ: вчера 04.09 14:00–19:00 локально)
    st.caption("Период: 04.09 с 14:00 до 19:00 (локальное время)")

    start_utc, end_utc = local_range_to_utc(DEFAULT_LOCAL_DATE, DEFAULT_START_HOUR, DEFAULT_END_HOUR)

    # Данные точки A (используем как “Собрано” по ТЗ)
    dfA = fetch_events("A", start_utc, end_utc)

    # Метрики
    collected_total = dfA["potato_id"].nunique() if not dfA.empty else 0
    st.metric("Собрано (шт)", value=f"{collected_total}")

    # Поток по часам (Собрано)
    st.markdown("### Поток по часам")
    hc = render_hour_chart_collected(dfA)

    # Таблица по категориям (мм)
    st.markdown("### Таблица по категориям (мм)")
    bins_df = bins_table_mm_collected(dfA)
    st.dataframe(bins_df, use_container_width=True)

    # Экспорт
    hour_export = hc.rename(columns={"hour": "Дата и час", "collected": "Собрано (шт)"}) if not hc.empty else pd.DataFrame(columns=["Дата и час","Собрано (шт)"])
    file_bytes, ext, mime = make_excel_bytes(hour_export, bins_df)
    st.download_button(
        label="Скачать отчёт",
        data=file_bytes,
        file_name=f"potato_report_2025-09-04_14-19.{ext}",
        mime=mime,
        use_container_width=True
    )

# ====== MAIN ======
def main():
    page_dashboard()

if __name__ == "__main__":
    main()
