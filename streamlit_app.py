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

# ===== Константы периода отображения =====
TZ = ZoneInfo("Asia/Aqtobe")  # только для внутренних конверсий
TARGET_DAY_LOCAL   = date(2025, 9, 4)  # "вчера"
TARGET_START_HOUR  = 14
TARGET_END_HOUR    = 19   # правая граница (итого 5 часов: 14..18)
LIVE_MINUTES       = 60   # берём последние 60 минут из БД

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
  .block-container { padding-top: 1.5rem; }
  .title-row { display:flex; align-items:center; gap:12px; }
  .title-row h1 { margin:0; font-size:26px; font-weight:800; letter-spacing:.3px; }
  .date-pill { padding:4px 10px; border-radius:999px; background:#f2f2f2; font-size:13px; }
  hr { margin: 10px 0 14px 0; opacity:.25; }
</style>
""", unsafe_allow_html=True)

def header():
    st.markdown(
        f"<div class='title-row'><h1>Система отслеживания и учёта картофеля</h1>"
        f"<span class='date-pill'>4 сентября</span></div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr/>", unsafe_allow_html=True)
    # Кнопка обновления — чуть ниже заголовка
    if st.button("↻ Обновить страницу", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===== Время/конверсии =====
def to_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def local_to_utc(d: date, hour: int) -> datetime:
    loc = datetime.combine(d, dtime(hour=hour)).replace(tzinfo=TZ)
    return loc.astimezone(timezone.utc)

def live_window_utc(minutes: int = LIVE_MINUTES) -> tuple[datetime, datetime]:
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    return now_utc - timedelta(minutes=minutes), now_utc

# ===== Чтение данных =====
@st.cache_data(ttl=5)
def fetch_events(point: Optional[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Возвращает ts (UTC-aware), point, potato_id, width_cm/height_cm или width/height (в см)."""
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

# ===== Подготовка размеров в миллиметрах =====
def add_mm_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Приоритет: *_cm; иначе width/height трактуем как см
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

# ===== Категории (мм) =====
CAT_LABELS = ["<30 мм", "30–40 мм", "40–50 мм", "50–60 мм", ">60 мм"]
CAT_BINS_MM = [0, 30, 40, 50, 60, 1_000_000]

def bins_table_mm_collected(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS)})
    d = add_mm_columns(df)
    if "width_mm" not in d.columns or d["width_mm"].dropna().empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS)})
    cut = pd.cut(d["width_mm"].fillna(-1), bins=CAT_BINS_MM, labels=CAT_LABELS, right=False, include_lowest=True)
    vc = cut.value_counts().reindex(CAT_LABELS).fillna(0).astype(int)
    return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [int(vc[c]) for c in CAT_LABELS]})

# ===== «Вчерашние» часы (фиксированная ось: 14..18) =====
def fixed_target_hours_index() -> pd.DatetimeIndex:
    # создаём 5 часов: 14:00, 15:00, 16:00, 17:00, 18:00 (локальное время)
    hours_local = [datetime.combine(TARGET_DAY_LOCAL, dtime(h)) for h in range(TARGET_START_HOUR, TARGET_END_HOUR)]
    # показывать красиво без TZ
    return pd.DatetimeIndex(pd.to_datetime(hours_local)).tz_localize(None)

# ===== Перенос live-данных в «вчерашнее» окно =====
def remap_live_to_target(df: pd.DataFrame, live_start_utc: datetime) -> pd.DataFrame:
    """
    Каждому событию присваиваем «виртуальный» ts_disp так:
    - все события из последних 60 минут отображаем в часе 14:00–15:00 вчера.
    Ось графика при этом фиксированная: 14..18, пустые часы = 0.
    """
    if df.empty or "ts" not in df.columns:
        return df
    # нормализуем к началу live-часа
    offs = (df["ts"] - live_start_utc).dt.total_seconds().clip(lower=0)
    # «виртуальная» точка старта: 4 сент 14:00 локально → UTC
    target_start_utc = local_to_utc(TARGET_DAY_LOCAL, TARGET_START_HOUR)
    df = df.copy()
    df["ts_disp"] = target_start_utc + pd.to_timedelta(offs, unit="s")
    return df

# ===== Агрегация «Собрано по часам» с заполнением пустых часов =====
def hour_counts_collected_fixed(df_disp: pd.DataFrame) -> pd.DataFrame:
    # База часов (локальные, без TZ для оси)
    base_hours = fixed_target_hours_index()
    if df_disp.empty or "ts_disp" not in df_disp.columns:
        return pd.DataFrame({"hour": base_hours, "Собрано (шт)": [0]*len(base_hours)})

    # для подсчёта переводим ts_disp в локальное → округляем до часа → без TZ
    ts_local = df_disp["ts_disp"].dt.tz_convert(TZ)
    hours_naive = ts_local.dt.floor("h").dt.tz_localize(None)
    g = (
        pd.DataFrame({"hour": hours_naive, "potato_id": df_disp["potato_id"]})
        .groupby("hour", as_index=False)
        .agg(**{"Собрано (шт)": ("potato_id", "nunique")})
    )
    # джоиним к базе часов, чтобы показать 0 там, где нет данных
    out = pd.DataFrame({"hour": base_hours}).merge(g, on="hour", how="left")
    out["Собрано (шт)"] = out["Собрано (шт)"].fillna(0).astype(int)
    return out

# ===== График =====
def render_hour_chart_fixed(df_disp: pd.DataFrame):
    hc = hour_counts_collected_fixed(df_disp)
    chart = (
        alt.Chart(hc)
        .mark_bar()
        .encode(
            x=alt.X("hour:T", title="Час"),
            y=alt.Y("Собрано (шт):Q", title="Собрано (шт)"),
            tooltip=[alt.Tooltip("hour:T", title="Час"), alt.Tooltip("Собрано (шт):Q")],
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

# ===== Калькулятор капитала (по категориям мм) =====
DEFAULT_WEIGHT_G = {"<30 мм": 20.0, "30–40 мм": 48.0, "40–50 мм": 83.0, "50–60 мм": 130.0, ">60 мм": 205.0}
DEFAULT_PRICE_KG = {"<30 мм": 0.0,  "30–40 мм": 0.0,  "40–50 мм": 0.0,  "50–60 мм": 0.0,  ">60 мм": 0.0}

def capital_calculator_mm(bins_df: pd.DataFrame):
    st.markdown("### Калькулятор капитала")
    counts = dict(zip(bins_df["Категория"], bins_df["Собрано (шт)"]))

    col_w = st.columns(5)
    col_p = st.columns(5)
    weights_g = {}
    prices_kg = {}

    for i, cat in enumerate(CAT_LABELS):
        with col_w[i]:
            weights_g[cat] = st.number_input(
                f"Вес ({cat}), г/шт",
                min_value=0.0,
                step=10.0,
                value=float(DEFAULT_WEIGHT_G.get(cat, 0.0)),
                format="%.2f",
                key=f"calc_w_{cat}",
            )
        with col_p[i]:
            prices_kg[cat] = st.number_input(
                f"Цена ({cat}), тг/кг",
                min_value=0.0,
                step=10.0,
                value=float(DEFAULT_PRICE_KG.get(cat, 0.0)),
                format="%.2f",
                key=f"calc_p_{cat}",
            )

    kg_totals = {cat: (counts.get(cat, 0) * weights_g.get(cat, 0.0)) / 1000.0 for cat in CAT_LABELS}
    subtotals = {cat: kg_totals[cat] * prices_kg.get(cat, 0.0) for cat in CAT_LABELS}
    total_sum = round(sum(subtotals.values()), 2)

    calc_df = pd.DataFrame({
        "Категория":        CAT_LABELS,
        "Собрано (шт)":     [int(counts.get(c, 0)) for c in CAT_LABELS],
        "Вес, г/шт":        [weights_g[c] for c in CAT_LABELS],
        "Итого, кг":        [round(kg_totals[c], 3) for c in CAT_LABELS],
        "Цена, тг/кг":      [prices_kg[c] for c in CAT_LABELS],
        "Сумма, тг":        [round(subtotals[c], 2) for c in CAT_LABELS],
    })
    st.dataframe(calc_df, use_container_width=True)
    st.subheader(f"Итого капитал: **{total_sum:,.2f} тг**".replace(",", " "))

# ===== Демонстрационная “Весовая таблица” =====
def render_weight_table(day: date):
    import random
    rng = random.Random(1000 + int(day.strftime("%Y%m%d")))
    hours = [10, 12, 14, 16]
    weights = [round(rng.uniform(0.12, 0.22), 3) for _ in hours]  # тонны
    rows = []
    for h, w in zip(hours, weights):
        ts = datetime.combine(day, dtime(h, 0))
        rows.append({"Дата и час": ts.strftime("%Y-%m-%d %H:%M"), "Вес, т": w})
    df = pd.DataFrame(rows)
    st.markdown("### Весовая таблица (демо)")
    st.dataframe(df, use_container_width=True)

# ===== Страница =====
def page_dashboard():
    header()

    # 1) Live-окно (последний час) → читаем точку A
    live_start_utc, live_end_utc = live_window_utc(LIVE_MINUTES)
    df_live = fetch_events("A", live_start_utc, live_end_utc)

    # 2) Переносим эти события в «вчерашний» 14:00–15:00
    df_disp = remap_live_to_target(df_live, live_start_utc)

    # 3) Метрика «Собрано (шт)»
    collected_total = df_disp["potato_id"].nunique() if not df_disp.empty else 0
    st.metric("Собрано (шт)", value=f"{collected_total}")

    # 4) Поток по часам (фиксированная ось 14..18 — дальше часы покажутся нулями)
    st.markdown("### Поток по часам")
    hc = render_hour_chart_fixed(df_disp)

    # 5) Таблица по категориям (мм)
    st.markdown("### Таблица по категориям (мм)")
    bins_df = bins_table_mm_collected(df_disp)
    st.dataframe(bins_df, use_container_width=True)

    # 6) Экспорт
    hour_export = hc.rename(columns={"hour": "Дата и час"}) if not hc.empty else pd.DataFrame(columns=["Дата и час","Собрано (шт)"])
    file_bytes, ext, mime = make_excel_bytes(hour_export, bins_df)
    st.download_button(
        label="Скачать отчёт",
        data=file_bytes,
        file_name=f"potato_report_2025-09-04_14-19.{ext}",
        mime=mime,
        use_container_width=True
    )

    # 7) Калькулятор капитала
    capital_calculator_mm(bins_df)

    # 8) Весовая таблица (демо)
    render_weight_table(TARGET_DAY_LOCAL)

# ===== MAIN =====
def main():
    page_dashboard()

if __name__ == "__main__":
    main()
