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

# ===== Настройки отображения =====
TZ = ZoneInfo("Asia/Aqtobe")     # локальное время для отображения
TARGET_DAY_LOCAL   = date(2025, 9, 4)  # показываем "вчера": 4 сентября
TARGET_START_HOUR  = 14                 # 14,15,16,17,18 (5 часов)
TARGET_END_HOUR    = 19                # не включительно

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
    if st.button("↻ Обновить", use_container_width=True):
        st.rerun()

# ===== Утилиты времени =====
def to_aware_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def local_to_utc(d: date, hour: int) -> datetime:
    return datetime.combine(d, dtime(hour=hour)).replace(tzinfo=TZ).astimezone(timezone.utc)

def fixed_target_hours_index() -> pd.DatetimeIndex:
    hours_local = [datetime.combine(TARGET_DAY_LOCAL, dtime(h))
                   for h in range(TARGET_START_HOUR, TARGET_END_HOUR)]
    return pd.DatetimeIndex(pd.to_datetime(hours_local)).tz_localize(None)

# ===== Чтение Supabase (последний batch, без фильтра точки) =====
def get_latest_batch() -> Optional[str]:
    if not USE_SUPABASE:
        return None
    try:
        r = _sb.table("events").select("batch,ts").order("ts", desc=True).limit(1).execute()
        if r.data and r.data[0].get("batch"):
            return r.data[0]["batch"]
    except Exception as e:
        st.warning(f"Не удалось получить latest batch: {e}")
    return None

def fetch_events_by_batch(batch: str) -> pd.DataFrame:
    if not USE_SUPABASE:
        return pd.DataFrame(columns=["ts","potato_id","width_cm","height_cm","width","height","batch"])
    try:
        rows = _sb.table("events").select("*").eq("batch", batch).order("ts", desc=False).execute().data
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        for c in ("width_cm","height_cm","width","height"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Ошибка чтения batch={batch}: {e}")
        return pd.DataFrame()

# ===== Размеры → миллиметры + категории =====
def add_mm_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # width_cm/height_cm приоритетны; если их нет — width/height (тоже в см)
    d["width_mm"]  = pd.to_numeric(d.get("width_cm", d.get("width")), errors="coerce") * 10.0
    d["height_mm"] = pd.to_numeric(d.get("height_cm", d.get("height")), errors="coerce") * 10.0
    return d

CAT_LABELS = ["<30 мм", "30–40 мм", "40–50 мм", "50–60 мм", ">60 мм"]
CAT_BINS_MM = [0, 30, 40, 50, 60, 1_000_000]

def bins_table_mm_collected(df: pd.DataFrame) -> pd.DataFrame:
    """Категории по ширине (мм) по уникальным potato_id."""
    if df.empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS)})
    d = add_mm_columns(df)
    # дедуп по клубню (берём последнюю запись по времени)
    if "potato_id" in d.columns:
        d = d.sort_values("ts").drop_duplicates(subset="potato_id", keep="last")
    if "width_mm" not in d.columns or d["width_mm"].dropna().empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS)})
    cut = pd.cut(d["width_mm"].fillna(-1), bins=CAT_BINS_MM, labels=CAT_LABELS,
                 right=False, include_lowest=True)
    vc = cut.value_counts().reindex(CAT_LABELS).fillna(0).astype(int)
    return pd.DataFrame({"Категория": CAT_LABELS,
                         "Собрано (шт)": [int(vc[c]) for c in CAT_LABELS]})

# ===== Перенос времени запуска → вчера 14–19 =====
def remap_run_to_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Переносим события текущего batch на шкалу «вчера 14–19»:
    час 0 от старта запуска -> 14–15, час 1 -> 15–16, ... (клип до 5 часов).
    """
    if df.empty or "ts" not in df.columns:
        return df
    d = df.copy()
    run_start = d["ts"].min()  # UTC, aware
    ts_floor  = d["ts"].dt.floor("h")
    elapsed_h = ((ts_floor - run_start) / pd.Timedelta(hours=1)).astype(int).clip(lower=0, upper=4)  # 0..4
    minute_off = d["ts"] - ts_floor
    target_start_utc = local_to_utc(TARGET_DAY_LOCAL, TARGET_START_HOUR)
    d["ts_disp"] = target_start_utc + pd.to_timedelta(elapsed_h, unit="h") + minute_off
    return d

def hour_counts_collected_fixed(df_disp: pd.DataFrame) -> pd.DataFrame:
    base_hours = fixed_target_hours_index()
    if df_disp.empty or "ts_disp" not in df_disp.columns:
        return pd.DataFrame({"hour": base_hours, "Собрано (шт)": [0]*len(base_hours)})
    ts_local = df_disp["ts_disp"].dt.tz_convert(TZ)
    hours_naive = ts_local.dt.floor("h").dt.tz_localize(None)
    g = (pd.DataFrame({"hour": hours_naive, "potato_id": df_disp["potato_id"]})
           .groupby("hour", as_index=False)
           .agg(**{"Собрано (шт)": ("potato_id", "nunique")}))
    out = pd.DataFrame({"hour": base_hours}).merge(g, on="hour", how="left")
    out["Собрано (шт)"] = out["Собрано (шт)"].fillna(0).astype(int)
    return out

def render_hour_chart_fixed(df_disp: pd.DataFrame):
    hc = hour_counts_collected_fixed(df_disp)
    chart = (
        alt.Chart(hc)
        .mark_bar()
        .encode(
            x=alt.X("hour:T", title="Час"),
            y=alt.Y("Собрано (шт):Q", title="Собрано (шт)"),
            tooltip=[alt.Tooltip("hour:T", title="Час"),
                     alt.Tooltip("Собрано (шт):Q")],
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
            hour_df.to_excel(w, index=False, sheet_name="Поток по часам (04.09)")
            bins_df.to_excel(w, index=False, sheet_name="Категории (мм)")
        return buf.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        pass
    try:
        import openpyxl  # noqa
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm") as w:
            hour_df.to_excel(w, index=False, sheet_name="Поток по часам (04.09)")
            bins_df.to_excel(w, index=False, sheet_name="Категории (мм)")
        return buf.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        pass
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hour_flow.csv", hour_df.to_csv(index=False))
        zf.writestr("bins_mm.csv",  bins_df.to_csv(index=False))
    return buf.getvalue(), "zip", "application/zip"

# ===== Калькулятор капитала (по умолчанию нули) =====
DEFAULT_WEIGHT_G = {c: 0.0 for c in CAT_LABELS}
DEFAULT_PRICE_KG = {c: 0.0 for c in CAT_LABELS}

def capital_calculator_mm(bins_df: pd.DataFrame):
    st.markdown("### Калькулятор капитала")
    counts = dict(zip(bins_df["Категория"], bins_df["Собрано (шт)"]))
    col_w = st.columns(5); col_p = st.columns(5)
    weights_g, prices_kg = {}, {}
    for i, cat in enumerate(CAT_LABELS):
        with col_w[i]:
            weights_g[cat] = st.number_input(
                f"Вес ({cat}), г/шт", min_value=0.0, step=1.0,
                value=float(DEFAULT_WEIGHT_G.get(cat, 0.0)), format="%.2f",
                key=f"calc_w_{cat}"
            )
        with col_p[i]:
            prices_kg[cat] = st.number_input(
                f"Цена ({cat}), тг/кг", min_value=0.0, step=1.0,
                value=float(DEFAULT_PRICE_KG.get(cat, 0.0)), format="%.2f",
                key=f"calc_p_{cat}"
            )
    kg_totals = {cat: (counts.get(cat, 0) * weights_g.get(cat, 0.0)) / 1000.0 for cat in CAT_LABELS}
    subtotals = {cat: kg_totals[cat] * prices_kg.get(cat, 0.0) for cat in CAT_LABELS}
    total_sum = round(sum(subtotals.values()), 2)
    calc_df = pd.DataFrame({
        "Категория":   CAT_LABELS,
        "Собрано (шт)": [int(counts.get(c, 0)) for c in CAT_LABELS],
        "Вес, г/шт":   [weights_g[c] for c in CAT_LABELS],
        "Итого, кг":   [round(kg_totals[c], 3) for c in CAT_LABELS],
        "Цена, тг/кг": [prices_kg[c] for c in CAT_LABELS],
        "Сумма, тг":   [round(subtotals[c], 2) for c in CAT_LABELS],
    })
    st.dataframe(calc_df, use_container_width=True)
    st.subheader(f"Итого капитал: **{total_sum:,.2f} тг**".replace(",", " "))

# ===== Страница =====
def page_dashboard():
    header()

    latest_batch = get_latest_batch()
    if not latest_batch:
        st.info("Нет данных (колонка batch пуста). Убедись, что приложение пишет batch для событий.")
        return

    # читаем весь последний batch
    df_run = fetch_events_by_batch(latest_batch)

    # перенос часов в «вчера 14–19»
    df_disp = remap_run_to_target(df_run)

    # метрика: Собрано (шт) по уникальным potato_id
    total_collected = df_run["potato_id"].nunique() if not df_run.empty else 0
    st.metric("Собрано (шт)", value=f"{total_collected}")

    # Поток по часам (фиксированные 14..18)
    st.markdown("### Поток по часам")
    hc = render_hour_chart_fixed(df_disp)

    # Таблица по категориям (мм), по уникальным potato_id
    st.markdown("### Таблица по категориям (мм)")
    bins_df = bins_table_mm_collected(df_run)
    st.dataframe(bins_df, use_container_width=True)

    # Экспорт
    hour_export = hc.rename(columns={"hour": "Дата и час"})
    file_bytes, ext, mime = make_excel_bytes(hour_export, bins_df)
    st.download_button(
        label="Скачать отчёт",
        data=file_bytes,
        file_name=f"potato_report_{latest_batch}.{ext}",
        mime=mime,
        use_container_width=True
    )

    # Калькулятор
    capital_calculator_mm(bins_df)

# ===== MAIN =====
def main():
    page_dashboard()

if __name__ == "__main__":
    main()
