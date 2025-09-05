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

# ===== Константы отображения =====
TZ = ZoneInfo("Asia/Aqtobe")   # для внутренних конверсий
TARGET_DAY_LOCAL   = date(2025, 9, 4)  # "вчера"
TARGET_START_HOUR  = 14                 # показываем 14,15,16,17,18 (5 часов)
TARGET_END_HOUR    = 19
LIVE_HOURS         = 5                  # сколько последних часов отображать (макс 5)

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
    if st.button("↻ Обновить страницу", use_container_width=True):
        st.rerun()
    # автообновление
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="auto_refresh_5s")
    except Exception:
        st.markdown("<meta http-equiv='refresh' content='10'>", unsafe_allow_html=True)

# ===== Время/конверсии =====
def to_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def local_to_utc(d: date, hour: int) -> datetime:
    loc = datetime.combine(d, dtime(hour=hour)).replace(tzinfo=TZ)
    return loc.astimezone(timezone.utc)

def window_last_hours_utc(hours: int) -> tuple[datetime, datetime]:
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    return now_utc - timedelta(hours=hours), now_utc

# ===== Чтение данных (без кэша) =====
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

# ===== Размеры → мм =====
def add_mm_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
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
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS]})
    d = add_mm_columns(df)
    if "width_mm" not in d.columns or d["width_mm"].dropna().empty:
        return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [0]*len(CAT_LABELS]})
    cut = pd.cut(d["width_mm"].fillna(-1), bins=CAT_BINS_MM, labels=CAT_LABELS, right=False, include_lowest=True)
    vc = cut.value_counts().reindex(CAT_LABELS).fillna(0).astype(int)
    return pd.DataFrame({"Категория": CAT_LABELS, "Собрано (шт)": [int(vc[c]) for c in CAT_LABELS]})

# ===== Фиксированная ось часов (вчера: 14..18) =====
def fixed_target_hours_index() -> pd.DatetimeIndex:
    hours_local = [datetime.combine(TARGET_DAY_LOCAL, dtime(h)) for h in range(TARGET_START_HOUR, TARGET_END_HOUR)]
    return pd.DatetimeIndex(pd.to_datetime(hours_local)).tz_localize(None)

# ===== Маппинг последних N часов в 14..19 (с сохранением минут/секунд внутри часа) =====
def remap_multi_live_hours_to_target(df: pd.DataFrame, now_utc: datetime, hours: int) -> pd.DataFrame:
    """
    Для каждого события ts:
      delta_h = floor((now_utc - ts) / 1h)
      если 0 <= delta_h < hours: переносим в TARGET_START_HOUR + delta_h (UTC), сохраняя offset внутри часа.
    """
    if df.empty or "ts" not in df.columns:
        return df

    df = df.copy()
    # вычисляем индекс live-часа (0 = текущий час, 1 = предыдущий, ...)
    delta = (now_utc - df["ts"]).dt.total_seconds()
    bin_idx = (delta // 3600).astype("Int64")  # допускаем NA
    mask = (bin_idx >= 0) & (bin_idx < hours)
    df = df[mask.fillna(False)].copy()
    if df.empty:
        return df

    # Смещение внутри своего часа (минуты/секунды) сохраняем
    ts_floor = df["ts"].dt.floor("h")
    minute_offset = df["ts"] - ts_floor

    target_start_utc = local_to_utc(TARGET_DAY_LOCAL, TARGET_START_HOUR)
    df["ts_disp"] = target_start_utc + pd.to_timedelta(bin_idx[mask].astype(int), unit="h") + minute_offset
    return df

# ===== Агрегация по часам (фиксированная ось) =====
def hour_counts_collected_fixed(df_disp: pd.DataFrame) -> pd.DataFrame:
    base_hours = fixed_target_hours_index()
    if df_disp.empty or "ts_disp" not in df_disp.columns:
        return pd.DataFrame({"hour": base_hours, "Собрано (шт)": [0]*len(base_hours)})

    ts_local = df_disp["ts_disp"].dt.tz_convert(TZ)
    hours_naive = ts_local.dt.floor("h").dt.tz_localize(None)
    g = (
        pd.DataFrame({"hour": hours_naive, "potato_id": df_disp["potato_id"]})
        .groupby("hour", as_index=False)
        .agg(**{"Собрано (шт)": ("potato_id", "nunique")})
    )
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

# ===== Калькулятор капитала (значения по умолчанию = 0) =====
CAT_LABELS = ["<30 мм", "30–40 мм", "40–50 мм", "50–60 мм", ">60 мм"]
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
                value=float(DEFAULT_WEIGHT_G.get(cat, 0.0)), format="%.2f", key=f"calc_w_{cat}"
            )
        with col_p[i]:
            prices_kg[cat] = st.number_input(
                f"Цена ({cat}), тг/кг", min_value=0.0, step=1.0,
                value=float(DEFAULT_PRICE_KG.get(cat, 0.0)), format="%.2f", key=f"calc_p_{cat}"
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

# ===== Демо “Весовая таблица” =====
def render_weight_table(day: date):
    import random
    rng = random.Random(1000 + int(day.strftime("%Y%m%d")))
    hours = [10, 12, 14, 16]
    weights = [round(rng.uniform(0.12, 0.22), 3) for _ in hours]
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

    # 1) читаем последние N часов из БД (точка A)
    start_utc, now_utc = window_last_hours_utc(LIVE_HOURS)
    df_live = fetch_events("A", start_utc, now_utc)

    # 2) переносим каждый live-час в «вчерашние» 14..19
    df_disp = remap_multi_live_hours_to_target(df_live, now_utc, LIVE_HOURS)

    # 3) Метрика «Собрано (шт)» — суммарно для всех видимых часов
    collected_total = df_disp["potato_id"].nunique() if not df_disp.empty else 0
    st.metric("Собрано (шт)", value=f"{collected_total}")

    # 4) Поток по часам (фиксированная ось 14..18)
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
