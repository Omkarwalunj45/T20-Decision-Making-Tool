# =============================================================================
# IPL Decision Intelligence Engine — Streamlit App
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import glob, json, itertools, ast, os, contextlib
from io import StringIO
from typing import List, Dict, Any
import warnings; warnings.filterwarnings("ignore")

try:
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Decision Intelligence Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --gold: #f4c430; --gold2: #e6b020;
    --text: #e6edf3; --text2: #8b949e;
    --border: #30363d; --success: #3fb950; --danger: #f85149;
    --blue: #58a6ff;
  }
  .stApp { background-color: var(--bg); color: var(--text); }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0d1117 100%);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }
  .app-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 50%, #0a0e1a 100%);
    border-bottom: 2px solid var(--gold);
    padding: 20px 32px 16px;
    margin: -1rem -1rem 1.5rem;
  }
  .app-header h1 {
    font-size: 2rem; font-weight: 700; color: var(--gold);
    letter-spacing: 0.02em; margin: 0;
  }
  .app-header .subtitle { color: var(--text2); font-size: 0.9rem; margin-top: 4px; }
  .match-banner {
    background: var(--bg2); border: 1px solid var(--border);
    border-left: 4px solid var(--gold); border-radius: 8px;
    padding: 12px 20px; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 16px;
  }
  .match-banner .team { font-size: 1.1rem; font-weight: 600; color: var(--text); }
  .match-banner .vs { color: var(--gold); font-weight: 700; font-size: 1.2rem; }
  .match-banner .venue { color: var(--text2); font-size: 0.85rem; }
  .metric-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 20px; text-align: center;
  }
  .metric-card .val { font-size: 1.8rem; font-weight: 700; color: var(--gold); }
  .metric-card .lbl { font-size: 0.8rem; color: var(--text2); margin-top: 4px; }
  .player-table { width: 100%; border-collapse: collapse; }
  .player-table th {
    background: var(--bg3); color: var(--text2); font-size: 0.78rem;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
    padding: 10px 14px; border-bottom: 1px solid var(--border); text-align: left;
  }
  .player-table td {
    padding: 9px 14px; border-bottom: 1px solid var(--border);
    color: var(--text); font-size: 0.88rem;
  }
  .player-table tr:hover td { background: var(--bg3); }
  .rank-badge {
    display: inline-block; width: 26px; height: 26px; border-radius: 50%;
    background: var(--bg3); color: var(--text2); font-size: 0.78rem;
    font-weight: 600; text-align: center; line-height: 26px;
  }
  .rank-1 { background: #c9a227; color: #0d1117; }
  .rank-2 { background: #8d9094; color: #0d1117; }
  .rank-3 { background: #8c5a2c; color: #e6edf3; }
  .role-badge {
    display: inline-block; padding: 2px 8px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500;
  }
  .role-bat    { background: #1f3d2a; color: #3fb950; border: 1px solid #2ea043; }
  .role-bowl   { background: #3d1f1f; color: #f85149; border: 1px solid #da3633; }
  .role-wk     { background: #1f2d3d; color: #58a6ff; border: 1px solid #388bfd; }
  .role-ar     { background: #3d2e1f; color: #f4c430; border: 1px solid #e6b020; }
  .section-title {
    font-size: 1.05rem; font-weight: 600; color: var(--text);
    border-bottom: 1px solid var(--border); padding-bottom: 8px; margin: 1.5rem 0 1rem;
  }
  .step-badge {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; color: var(--text2);
    display: inline-block; margin-right: 8px;
  }
  .step-done { border-color: var(--success); color: var(--success); }
  .step-pending { border-color: var(--border); color: var(--text2); }
  div[data-testid="stHorizontalBlock"] { gap: 12px; }
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg2); border-bottom: 1px solid var(--border);
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--text2) !important;
    border-radius: 6px 6px 0 0; padding: 8px 16px;
    font-size: 0.85rem; font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: var(--bg) !important; color: var(--gold) !important;
    border-bottom: 2px solid var(--gold);
  }
  .stButton > button {
    background: var(--gold); color: #0d1117; font-weight: 600;
    border: none; border-radius: 6px; padding: 8px 20px; cursor: pointer;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: var(--gold2); }
  .stSelectbox > div, .stMultiSelect > div {
    background: var(--bg2) !important; border-color: var(--border) !important;
    color: var(--text) !important;
  }
  .stTextInput > div > input {
    background: var(--bg2) !important; border-color: var(--border) !important;
    color: var(--text) !important;
  }
  [data-testid="stDataFrame"] { background: var(--bg2); border-radius: 8px; }
  .stExpander { border: 1px solid var(--border) !important; border-radius: 8px !important; }
  .stExpander header { color: var(--text2) !important; }
  .status-ok  { color: var(--success); font-weight: 600; }
  .status-err { color: var(--danger);  font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
PREDEFINED_ICR = {
    "ramandeep singh": 71, "sanvir singh": 49.99, "jamie smith": 60.05,
    "dewald brevis": 81.30, "ayush mhatre": 79.95, "wanindu hasaranga": 17.94,
    "shubham dubey": 59.99, "matt henry_bowl": 88.59, "allah ghazanfar_bowl": 55.89,
    "ramandeep singh_bowl": 0.0, "ravindra jadeja": 32.22, "ravindra jadeja_bowl": 70.60,
    "varun chakaravarthy": 97.31, "varun chakaravarthy_bowl": 91.31,
    "matheesha pathirana_bowl": 75.31, "jacob duffy": 70.30, "harshit rana": 58.53,
    "ravi bishnoi": 58.53, "ravi bishnoi_bowl": 68.53, "cameron green": 88,
    "cameron green_bowl": 44.64, "mangesh yadav_bowl": 44.15, "prashant veer": 46,
    "prashant veer_bowl": 41.15, "m shahrukh khan": 43.28, "glenn philips": 49.89,
    "mohammad shami_bowl": 73, "aiden markram": 83, "nicholas pooran": 92,
    "tejasvi dahiya": 66, "sunil narine_bowl": 94, "pat cummins": 56,
    "pat cummins_bowl": 70, "liam livingstone": 71, "aniket verma": 70,
    "harshal patel_bowl": 73, "jaydev unadkat_bowl": 76, "heinrich klaasen": 92,
    "ishan kishan": 89, "kartik sharma": 55, "deepak chahar": 50,
    "hardik pandya": 95, "ashwani kumar_bowl": 78, "rohit sharma": 91,
    "mitchell santner_bowl": 83, "ben duckett": 83, "nitish rana": 83,
    "david miller": 90, "axar patel": 92
}

DEFAULT_PLAYERS_TO_REMOVE = [
    "nathan ellis", "mustafizur rahman", "ben duckett",
    "mitchell santner", "will jacks", "akash deep",
    "matheesha pathirana", "harshit rana"
]

SPIN_STYLES = ['SLA','OB','LBG','LB','LWS','OB/LB','OB/LBG','OB/SLA','SLA/LWS','LS']
PACE_STYLES = ['RMF','RF','RFM','LFM','LF','LMF']

ROLE_POS_LIMITS = {
    'top order batter':    (1, 5),
    'opener':              (1, 3),
    'opening batter':      (1, 3),
    'wicketkeeper batter': (1, 5),
    'batter':              (1, 6),
    'middle order batter': (3, 7),
    'lower order batter':  (5, 8),
    'batting allrounder':  (1, 8),
    'allrounder':          (1, 8),
    'bowling allrounder':  (8, 8),
    'bowler':              (8, 8),
}

BAT_SHARE_MAP = {1:0.221931,2:0.196456,3:0.123035,4:0.172610,
                 5:0.133640,6:0.082180,7:0.044305,8:0.025844}

ROLE_TO_BOWL_SHARE = {
    'bowler':0.200,'bowling allrounder':0.200,
    'allrounder':0.15,'batting allrounder':0.05
}

VENUE_TEXT = """venue\tvenue_factor\tspin_index
0\tRiverside Ground, Chester-le-Street\t0.941132\t0.376923
214\tOld Trafford, Manchester\t1.017011\t0.364583
423\tEdgbaston, Birmingham\t0.893010\t0.374101
663\tTrent Bridge, Nottingham\t0.974493\t0.374372
2623\tWankhede Stadium, Mumbai\t1.035000\t0.364362
2872\tMaharashtra Cricket Association Stadium, Pune\t1.102648\t0.288344
3129\tSaurashtra Cricket Association Stadium, Rajkot\t1.001152\t0.363636
3364\tJSCA International Stadium Complex, Ranchi\t1.042587\t0.411765
3616\tBharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow\t0.960182\t0.360788
3866\tNarendra Modi Stadium, Ahmedabad\t1.062830\t0.282787
14595\tM Chinnaswamy Stadium, Bengaluru\t0.983986\t0.273171
26556\tArun Jaitley Stadium, Delhi\t1.139000\t0.367470
26804\tRajiv Gandhi International Stadium, Uppal, Hyderabad\t1.002000\t0.322581
27057\tEden Gardens, Kolkata\t0.981000\t0.351097
27260\tMA Chidambaram Stadium, Chepauk, Chennai\t0.931000\t0.318584
33949\tPunjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh\t1.178140\t0.396104
34074\tHolkar Cricket Stadium, Indore\t1.050676\t0.306452
92206\tSawai Mansingh Stadium, Jaipur\t0.971000\t0.434555
174840\tBarsapara Cricket Stadium, Guwahati\t0.923532\t0.333333
177650\tMaharaja Yadavindra Singh International Cricket Stadium, Mullanpur\t0.903645\t0.240506
181585\tHimachal Pradesh Cricket Association Stadium, Dharamsala\t1.025331\t0.250000
225418\tDr DY Patil Sports Academy, Mumbai\t0.979021\t0.219512"""

# ─── HELPER ───────────────────────────────────────────────────────────────────
def capture(fn, *args, **kwargs):
    """Run fn(*args,**kwargs), capture stdout, return (result, logs)."""
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()

def role_badge(role):
    role = str(role).lower()
    if "wicket" in role: css = "role-wk"
    elif "allrounder" in role or "all-rounder" in role: css = "role-ar"
    elif "bowler" in role or "bowling" in role: css = "role-bowl"
    else: css = "role-bat"
    return f'<span class="role-badge {css}">{role.title()}</span>'

def prob_bar(val, max_val=100):
    pct = min(max(val / max_val * 100, 0), 100)
    color = "#3fb950" if pct > 60 else "#f4c430" if pct > 35 else "#8b949e"
    return f'<div style="background:#21262d;border-radius:3px;height:6px;width:100%;"><div style="background:{color};height:6px;border-radius:3px;width:{pct:.0f}%;"></div></div>'

# ─── ANALYSIS FUNCTIONS (logic unchanged from notebook) ───────────────────────

def role_flags(role):
    role = str(role)
    return pd.Series({
        "is_wk":    "wicketkeeper" in role,
        "is_open":  "opening" in role,
        "is_top":   "top order" in role,
        "is_middle":"middle order" in role,
        "is_bat":   any(k in role for k in ["batter","batting"]),
        "is_bowl":  "bowler" in role or "bowling" in role,
        "is_ar":    "allrounder" in role
    })

def optimize_team(team_name, df_sel):
    if not PULP_AVAILABLE:
        raise RuntimeError("PuLP not installed")
    team_df = df_sel[df_sel["Team"] == team_name].reset_index(drop=True)
    n = len(team_df)
    prob = LpProblem("team_select", LpMaximize)
    x = LpVariable.dicts("x", range(n), cat="Binary")
    prob += lpSum(team_df.loc[i,"baseline_prob"] * x[i] for i in range(n))
    prob += lpSum(x[i] for i in range(n)) == 12
    prob += lpSum(team_df.loc[i,"is_wk"] * x[i] for i in range(n)) >= 1
    prob += lpSum(team_df.loc[i,"is_bat"] * x[i] for i in range(n)) >= 7
    prob += lpSum(team_df.loc[i,"is_bowl"] * x[i] for i in range(n)) >= 6
    prob.solve(PULP_CBC_CMD(msg=0))
    selected = team_df[[x[i].value() == 1 for i in range(n)]].copy()
    return selected.sort_values("baseline_prob", ascending=False).reset_index(drop=True)

def build_team_form_adjusted_prob(our_team, opposition, venue, result_df, form_df):
    context = {"our_team": our_team, "opposition": opposition, "venue": venue}
    base = result_df.copy()
    base["Team"] = base["Team"].astype(str).str.strip()
    base["Player"] = base["Player"].astype(str).str.lower().str.strip()
    base["Role"] = base["Role"].astype(str).str.lower().str.strip()
    base["baseline_prob"] = pd.to_numeric(base["baseline_prob"], errors="coerce").fillna(0) * 100
    team_out = base[base["Team"] == our_team].copy()
    if team_out.empty:
        raise ValueError(f"No players found for team: {our_team}")
    team_players = set(team_out["Player"].tolist())
    fdf = form_df.copy()
    fdf["bat"] = fdf["bat"].astype(str).str.lower().str.strip()
    fdf["bowl"] = fdf["bowl"].astype(str).str.lower().str.strip()
    fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
    fdf["p_match"] = pd.to_numeric(fdf["p_match"], errors="coerce")
    if "bat_fantasy_pts" not in fdf.columns: fdf["bat_fantasy_pts"] = 0.0
    if "ball_fantasy_pts" not in fdf.columns: fdf["ball_fantasy_pts"] = 0.0
    fdf["bat_fantasy_pts"] = pd.to_numeric(fdf["bat_fantasy_pts"], errors="coerce").fillna(0.0)
    fdf["ball_fantasy_pts"] = pd.to_numeric(fdf["ball_fantasy_pts"], errors="coerce").fillna(0.0)
    sort_cols = [c for c in ["date","p_match","inns","ball_id"] if c in fdf.columns]
    fdf = fdf.sort_values(sort_cols).copy() if sort_cols else fdf.copy()

    def role_to_shares(role):
        role = str(role).lower().strip()
        if role in {"batter","wicketkeeper batter","top order batter","middle order batter","opening batter"}: return 1.0, 0.0
        if role == "bowler": return 0.0, 1.0
        if role == "allrounder": return 0.5, 0.5
        if role == "bowling allrounder": return 0.4, 0.6
        if role == "batting allrounder": return 0.6, 0.4
        return 0.5, 0.5

    def last_unique_matches(player):
        br = fdf.loc[fdf["bat"]==player, ["p_match","date"]].dropna().drop_duplicates()
        wr = fdf.loc[fdf["bowl"]==player, ["p_match","date"]].dropna().drop_duplicates()
        all_r = pd.concat([br,wr],ignore_index=True).drop_duplicates()
        if all_r.empty: return []
        return all_r.sort_values(["date","p_match"])["p_match"].drop_duplicates().tail(5).tolist()

    detail_rows = []
    for player in sorted(team_players):
        player_role = team_out.loc[team_out["Player"]==player,"Role"].iloc[0]
        bat_share, bowl_share = role_to_shares(player_role)
        recent = last_unique_matches(player)
        n_recent = len(recent)
        if n_recent < 3:
            detail_rows.append(pd.DataFrame([{"Team":our_team,"Player":player,"Role":player_role,
                "p_match":np.nan,"date":pd.NaT,"form_matches":n_recent,"bat_share":bat_share,
                "bowl_share":bowl_share,"bat_form_match":0.0,"bowl_form_match":0.0,
                "total_form_match":0.0,"eligible_for_form":False}]))
            continue
        pr = fdf[fdf["p_match"].isin(recent)].copy()
        bm = pr[pr["bat"]==player].groupby("p_match",as_index=False).agg(date=("date","max"),bat_form_match=("bat_fantasy_pts","sum"))
        wm = pr[pr["bowl"]==player].groupby("p_match",as_index=False).agg(date=("date","max"),bowl_form_match=("ball_fantasy_pts","sum"))
        pm = pd.merge(bm[["p_match","date","bat_form_match"]],wm[["p_match","bowl_form_match"]],on="p_match",how="outer")
        if "date" not in pm.columns: pm["date"] = pd.NaT
        pm["bat_form_match"] = pm["bat_form_match"].fillna(0.0)
        pm["bowl_form_match"] = pm["bowl_form_match"].fillna(0.0)
        pm["total_form_match"] = bat_share*pm["bat_form_match"] + bowl_share*pm["bowl_form_match"]
        pm["Team"]=our_team; pm["Player"]=player; pm["Role"]=player_role
        pm["form_matches"]=n_recent; pm["bat_share"]=bat_share; pm["bowl_share"]=bowl_share
        pm["eligible_for_form"]=True
        detail_rows.append(pm[["Team","Player","Role","p_match","date","form_matches","bat_share","bowl_share","bat_form_match","bowl_form_match","total_form_match","eligible_for_form"]])
    player_match_detail = pd.concat(detail_rows, ignore_index=True, sort=False)
    eligible_detail = player_match_detail[player_match_detail["eligible_for_form"]==True].copy()
    if eligible_detail.empty:
        summary = pd.DataFrame(columns=["Player","form_matches","bat_form_points","bowl_form_points","total_points"])
        eligible_summary = pd.DataFrame(columns=["Player","total_percentile","form_bonus"])
    else:
        summary = eligible_detail.groupby("Player",as_index=False).agg(
            form_matches=("p_match","nunique"),bat_form_points=("bat_form_match","sum"),
            bowl_form_points=("bowl_form_match","sum"),total_points=("total_form_match","sum"))
        eligible_summary = summary[summary["form_matches"]>=3].copy()
        if not eligible_summary.empty:
            eligible_summary["total_percentile"] = eligible_summary["total_points"].rank(pct=True)*100.0
            eligible_summary["form_bonus"] = 2.75*np.tanh((eligible_summary["total_percentile"]-50.0)/25.0)
            eligible_summary = eligible_summary[["Player","total_percentile","form_bonus"]]
        else:
            eligible_summary["total_percentile"] = pd.Series(dtype=float)
            eligible_summary["form_bonus"] = pd.Series(dtype=float)
    team_out = team_out.merge(summary[["Player","form_matches","bat_form_points","bowl_form_points","total_points"]],on="Player",how="left")
    team_out = team_out.merge(eligible_summary,on="Player",how="left")
    team_out["form_matches"] = team_out["form_matches"].fillna(0).astype(int)
    team_out["bat_form_points"] = team_out["bat_form_points"].fillna(0.0)
    team_out["bowl_form_points"] = team_out["bowl_form_points"].fillna(0.0)
    team_out["total_points"] = team_out["total_points"].fillna(np.nan)
    team_out["total_percentile"] = team_out.get("total_percentile", pd.Series(np.nan, index=team_out.index)).fillna(np.nan)
    team_out["form_bonus"] = team_out.get("form_bonus", pd.Series(0.0, index=team_out.index)).fillna(0.0)
    team_out["adjusted_prob_before_form"] = team_out["baseline_prob"]
    team_out["adjusted_prob"] = np.clip(team_out["baseline_prob"]+team_out["form_bonus"],0,100)
    team_out = team_out.sort_values("adjusted_prob",ascending=False).reset_index(drop=True)
    team_out["adjusted_rank"] = np.arange(1,len(team_out)+1)
    return team_out, player_match_detail, context

def get_similar_venues_with_closeness(venue_name, venue_lookup_df, threshold_factor=0.0495, threshold_spin=0.078080):
    unique_venues = venue_lookup_df.copy()
    target = unique_venues[unique_venues["venue"]==venue_name]
    if target.empty: return pd.DataFrame()
    tf = target["venue_factor"].iloc[0]; ts = target["spin_index"].iloc[0]
    df2 = unique_venues.copy()
    df2["difference"] = np.sqrt((df2["venue_factor"]-tf)**2+(df2["spin_index"]-ts)**2)
    mask = (np.abs(df2["venue_factor"]-tf)<=threshold_factor)&(np.abs(df2["spin_index"]-ts)<=threshold_spin)
    similar = df2[mask&(df2["venue"]!=venue_name)].copy()
    return similar.sort_values("difference").reset_index(drop=True)

def build_similar_venue_dataset(target_venue, mega_df, venue_df):
    similar_df = get_similar_venues_with_closeness(target_venue, venue_df)
    if similar_df.empty: return similar_df
    mc = mega_df.groupby("ground")["p_match"].nunique().reset_index().rename(columns={"ground":"venue","p_match":"matches_played"})
    similar_df = similar_df.merge(mc,on="venue",how="left")
    similar_df["matches_played"] = similar_df["matches_played"].fillna(0).astype(int)
    return similar_df

def apply_venue_adjustment_role_weighted(team_adjusted, mega_df, similar_venue_dataset, target_venue, alpha=1.5, beta=25.0, min_matches=3, print_details=False):
    info = team_adjusted.copy(); mega = mega_df.copy(); sim = similar_venue_dataset.copy()
    info["Player"] = info["Player"].astype(str).str.lower().str.strip()
    info["Role"] = info["Role"].astype(str).str.lower().str.strip()
    info["adjusted_prob"] = pd.to_numeric(info["adjusted_prob"],errors="coerce").fillna(0.0)
    mega["ground"] = mega["ground"].astype(str).str.strip()
    mega["bat"] = mega["bat"].astype(str).str.lower().str.strip()
    mega["bowl"] = mega["bowl"].astype(str).str.lower().str.strip()
    for c in ["p_match","batruns","ballfaced","bowlruns","out","score"]:
        mega[c] = pd.to_numeric(mega[c],errors="coerce").fillna(0)
    mega["control"] = pd.to_numeric(mega.get("control",np.nan),errors="coerce")
    similar_list = sim["venue"].astype(str).str.strip().tolist()
    venue_list = sorted(set(similar_list+[str(target_venue).strip()]))
    venue_slice = mega[mega["ground"].isin(venue_list)].copy()
    if venue_slice.empty: raise ValueError("venue_slice empty")
    def role_to_shares(role):
        role=str(role).lower().strip()
        if role in {"batter","opening batter","top order batter","middle order batter","wicketkeeper batter"}: return 1.0,0.0
        if role=="bowler": return 0.0,1.0
        if role=="allrounder": return 0.5,0.5
        if role=="bowling allrounder": return 0.4,0.6
        if role=="batting allrounder": return 0.6,0.4
        return 0.5,0.5
    player_rows=[]
    for player in sorted(set(info["Player"].tolist())):
        role_row = info.loc[info["Player"]==player,"Role"]
        role = role_row.iloc[0] if not role_row.empty else "batter"
        bs,ws = role_to_shares(role)
        tv = info.loc[info["Player"]==player,"Team"].iloc[0] if "Team" in info.columns else None
        pvr = venue_slice[(venue_slice["bat"]==player)|(venue_slice["bowl"]==player)].copy()
        if pvr.empty:
            player_rows.append({"Team":tv,"Player":player,"Role":role,"venue_matches":0,"bat_share":bs,"bowl_share":ws,"bat_venue_points":0.0,"bowl_venue_points":0.0,"total_points":0.0,"eligible_for_venue":False})
            continue
        bg = pvr[pvr["bat"]==player].copy()
        if not bg.empty:
            bm=int(bg["p_match"].nunique()); balls=float(bg["ballfaced"].sum()); runs=float(bg["batruns"].sum())
            outs=float(bg["out"].sum()); ctrl=float(bg["control"].mean()) if bg["control"].notna().any() else 0.0
            br=float(bg["batruns"].isin([4,6]).sum()/len(bg)); rr=runs/balls if balls>0 else 0.0; dr=outs/balls if balls>0 else 0.0
            bat_pts=(0.45*min(rr/6.0,1.0)+0.25*ctrl+0.20*br-0.10*dr)*bm
        else: bat_pts=0.0; bm=0
        wg = pvr[pvr["bowl"]==player].copy()
        if not wg.empty:
            wm=int(wg["p_match"].nunique()); bb=float(wg["ballfaced"].sum()); rc=float(wg["bowlruns"].sum())
            wk=float(wg["out"].sum()); ctb=float(wg["control"].mean()) if wg["control"].notna().any() else 0.0
            bwb=float(wg["batruns"].isin([4,6]).sum()/len(wg)); db=float((wg["score"]==0).sum()/len(wg))
            rpb=rc/bb if bb>0 else 0.0; wr=wk/bb if bb>0 else 0.0
            bowl_pts=(0.45*(1-min(rpb/6.0,1.0))+0.25*min(wr,1.0)+0.20*db+0.10*(1-bwb))*wm
        else: bowl_pts=0.0; wm=0
        tm=max(bm,wm); tp=bs*bat_pts+ws*bowl_pts
        player_rows.append({"Team":tv,"Player":player,"Role":role,"venue_matches":tm,"bat_share":bs,"bowl_share":ws,"bat_venue_points":bat_pts,"bowl_venue_points":bowl_pts,"total_points":tp,"eligible_for_venue":tm>=min_matches})
    stats=pd.DataFrame(player_rows)
    stats["venue_adjust"]=0.0; stats["total_percentile"]=np.nan
    eligible=stats[stats["eligible_for_venue"]].copy()
    if not eligible.empty:
        eligible=eligible.sort_values("total_points").reset_index(drop=True)
        eligible["total_percentile"]=eligible["total_points"].rank(pct=True)*100.0
        eligible["venue_adjust"]=alpha*np.tanh((eligible["total_percentile"]-50.0)/beta)
        pm=eligible.set_index("Player")["total_percentile"]; va=eligible.set_index("Player")["venue_adjust"]
        stats["total_percentile"]=stats["Player"].map(pm); stats["venue_adjust"]=stats["Player"].map(va).fillna(0.0)
    stale=["venue_matches","bat_share","bowl_share","bat_venue_points","bowl_venue_points","total_points","total_percentile","venue_adjust","adjusted_prob_before_venue","adjusted_rank"]
    info=info.drop(columns=[c for c in stale if c in info.columns],errors="ignore")
    sm=stats.set_index("Player")
    for col,fill in [("venue_matches",0),("bat_share",0.0),("bowl_share",0.0),("bat_venue_points",0.0),("bowl_venue_points",0.0),("total_points",0.0)]:
        info[col]=info["Player"].map(sm[col]).fillna(fill)
    info["total_percentile"]=info["Player"].map(sm["total_percentile"])
    info["venue_adjust"]=info["Player"].map(sm["venue_adjust"]).fillna(0.0)
    info["adjusted_prob_before_venue"]=info["adjusted_prob"]
    info["adjusted_prob"]=np.clip(info["adjusted_prob"]+info["venue_adjust"],0,100)
    info=info.sort_values("adjusted_prob",ascending=False).reset_index(drop=True)
    info["adjusted_rank"]=np.arange(1,len(info)+1)
    pvd=info[["Team","Player","Role","venue_matches","bat_share","bowl_share","bat_venue_points","bowl_venue_points","total_points","total_percentile","adjusted_prob_before_venue","venue_adjust","adjusted_prob","adjusted_rank"]].copy()
    return info, stats, pvd, venue_slice

def apply_matchup_adjustment(team_adjusted, df, our_team, opposition_xi, alpha=1.25, min_balls=5, print_pairs=False):
    info=team_adjusted.copy(); data=df.copy()
    info["Player"]=info["Player"].astype(str).str.lower().str.strip()
    info["Role"]=info["Role"].astype(str).str.lower().str.strip()
    for c in ["bat","bowl"]: data[c]=data[c].astype(str).str.lower().str.strip()
    data["batruns"]=pd.to_numeric(data.get("batruns",data.get("score",0)),errors="coerce").fillna(0)
    data["score"]=pd.to_numeric(data.get("score",0),errors="coerce").fillna(0)
    data["out"]=pd.to_numeric(data.get("out",0),errors="coerce").fillna(0)
    data["control"]=pd.to_numeric(data.get("control",np.nan),errors="coerce")
    info=info[info["Team"].astype(str).str.strip()==our_team].copy()
    if info.empty: raise ValueError(f"No players for team: {our_team}")
    opp_players=list(dict.fromkeys([str(v).lower().strip() for _,v in sorted(opposition_xi.items())]))
    info["adjusted_prob"]=pd.to_numeric(info["adjusted_prob"],errors="coerce").fillna(0.0)
    info["adjusted_prob_before_matchup"]=info["adjusted_prob"]
    bat_roles={"batter","opening batter","top order batter","middle order batter","wicketkeeper batter","batting allrounder","allrounder"}
    bowl_roles={"bowler","bowling allrounder","allrounder","Bowler"}
    our_batters=info[info["Role"].isin(bat_roles)]["Player"].dropna().unique().tolist()
    our_bowlers=info[info["Role"].isin(bowl_roles)]["Player"].dropna().unique().tolist()
    def bat_ps(pdf):
        b=len(pdf); r=pdf["batruns"].sum(); d=pdf["out"].sum()
        cr=pdf["control"].mean() if "control" in pdf.columns and pdf["control"].notna().any() else 0.0
        br=pdf["batruns"].isin([4,6]).sum()/b if b>0 else 0.0
        rr=r/b if b>0 else 0.0; dr=d/b if b>0 else 0.0
        return {"balls":b,"runs":r,"dismissals":d,"control_rate":cr,"boundary_rate":br,"run_rate":rr,"score":0.40*min(rr/6,1)+0.20*cr+0.20*br-0.20*dr}
    def bowl_ps(pdf):
        b=len(pdf); rc=pdf["batruns"].sum(); wk=pdf["out"].sum()
        cr=pdf["control"].mean() if "control" in pdf.columns and pdf["control"].notna().any() else 0.0
        br=pdf["batruns"].isin([4,6]).sum()/b if b>0 else 0.0
        dr=(pdf["score"]==0).sum()/b if b>0 else 0.0; rpb=rc/b if b>0 else 0.0; wr=wk/b if b>0 else 0.0
        return {"balls":b,"runs_conceded":rc,"wickets":wk,"control_rate":cr,"boundary_rate":br,"dot_rate":dr,"runs_per_ball":rpb,"score":0.45*(1-min(rpb/6,1))+0.25*min(wr,1)+0.20*dr+0.10*(1-br)}
    bpr=[]; wpr=[]
    for batter in our_batters:
        for op in opp_players:
            pdf=data[(data["bat"]==batter)&(data["bowl"]==op)].copy()
            if len(pdf)<min_balls: continue
            s=bat_ps(pdf); bpr.append({"player":batter,"opposition_player":op,"balls":s["balls"],"runs":s["runs"],"dismissals":s["dismissals"],"control_rate":s["control_rate"],"boundary_rate":s["boundary_rate"],"run_rate":s["run_rate"],"pair_score":s["score"]})
    for bowler in our_bowlers:
        for op in opp_players:
            pdf=data[(data["bowl"]==bowler)&(data["bat"]==op)].copy()
            if len(pdf)<min_balls: continue
            s=bowl_ps(pdf); wpr.append({"player":bowler,"opposition_player":op,"balls":s["balls"],"runs_conceded":s["runs_conceded"],"wickets":s["wickets"],"control_rate":s["control_rate"],"boundary_rate":s["boundary_rate"],"dot_rate":s["dot_rate"],"runs_per_ball":s["runs_per_ball"],"pair_score":s["score"]})
    bdf=pd.DataFrame(bpr); wdf=pd.DataFrame(wpr)
    if bdf.empty: bs=pd.DataFrame(columns=["Player","bat_matchup_points","bat_matchup_pairs","bat_matchup_percentile"])
    else:
        bs=bdf.groupby("player",as_index=False).agg(bat_matchup_points=("pair_score","sum"),bat_matchup_pairs=("pair_score","count")).rename(columns={"player":"Player"})
        bs["bat_matchup_percentile"]=bs["bat_matchup_points"].rank(pct=True)*100
    if wdf.empty: ws=pd.DataFrame(columns=["Player","bowl_matchup_points","bowl_matchup_pairs","bowl_matchup_percentile"])
    else:
        ws=wdf.groupby("player",as_index=False).agg(bowl_matchup_points=("pair_score","sum"),bowl_matchup_pairs=("pair_score","count")).rename(columns={"player":"Player"})
        ws["bowl_matchup_percentile"]=ws["bowl_matchup_points"].rank(pct=True)*100
    info=info.merge(bs,on="Player",how="left").merge(ws,on="Player",how="left")
    for col in ["bat_matchup_points","bat_matchup_pairs","bat_matchup_percentile","bowl_matchup_points","bowl_matchup_pairs","bowl_matchup_percentile"]:
        if col not in info.columns: info[col]=np.nan
    info["bat_matchup_points"]=info["bat_matchup_points"].fillna(0.0)
    info["bowl_matchup_points"]=info["bowl_matchup_points"].fillna(0.0)
    info["bat_matchup_pairs"]=info["bat_matchup_pairs"].fillna(0).astype(int)
    info["bowl_matchup_pairs"]=info["bowl_matchup_pairs"].fillna(0).astype(int)
    info["bat_share"]=pd.to_numeric(info.get("bat_share",0.5),errors="coerce").fillna(0.5)
    info["bowl_share"]=pd.to_numeric(info.get("bowl_share",0.5),errors="coerce").fillna(0.5)
    info["matchup_points_raw"]=info["bat_share"]*info["bat_matchup_points"]+info["bowl_share"]*info["bowl_matchup_points"]
    info["matchup_percentile"]=info["matchup_points_raw"].rank(pct=True)*100
    info["matchup_adjust"]=alpha*np.tanh((info["matchup_percentile"]-50)/25)
    info["adjusted_prob_before_matchup"]=info["adjusted_prob"]
    info["adjusted_prob"]=np.clip(info["adjusted_prob"]+info["matchup_adjust"],0,100)
    info=info.sort_values("adjusted_prob",ascending=False).reset_index(drop=True)
    info["adjusted_rank"]=np.arange(1,len(info)+1)
    md=info[["Team","Player","Role","bat_share","bowl_share","bat_matchup_points","bat_matchup_pairs","bat_matchup_percentile","bowl_matchup_points","bowl_matchup_pairs","bowl_matchup_percentile","matchup_points_raw","matchup_percentile","matchup_adjust","adjusted_prob_before_matchup","adjusted_prob","adjusted_rank"]].copy()
    return info, md, bdf, wdf

def apply_opposition_adjustment(team_adjusted, ipl_df, our_team, opposition_team, alpha=1.25, min_balls=20, print_details=False):
    info=team_adjusted.copy(); df=ipl_df.copy()
    for col in ["bat","bowl","team_bat","team_bowl"]: df[col]=df[col].astype(str).str.lower().str.strip()
    info["Player"]=info["Player"].astype(str).str.lower().str.strip()
    info["Role"]=info["Role"].astype(str).str.lower().str.strip()
    our=our_team.lower().strip(); opp=opposition_team.lower().strip()
    h2h=df[((df["team_bat"]==our)&(df["team_bowl"]==opp))|((df["team_bat"]==opp)&(df["team_bowl"]==our))].copy()
    if h2h.empty:
        print("No H2H data. Skipping opposition adjustment.")
        return info, pd.DataFrame()
    db=h2h[(h2h["team_bat"]==our)&(h2h["team_bowl"]==opp)]
    dw=h2h[(h2h["team_bowl"]==our)&(h2h["team_bat"]==opp)]
    bat_stats=[]
    for player in info["Player"]:
        pdf=db[db["bat"]==player]; balls=len(pdf)
        if balls<min_balls: continue
        runs=np.nansum(pdf["batruns"]); dismissals=np.nansum(pdf["out"])
        control=np.nanmean(pdf["control"]) if "control" in pdf.columns else 0
        boundary=(pdf["batruns"].isin([4,6])).sum()/balls
        rr=runs/balls; dr=dismissals/balls
        sc=0.40*min(rr/6,1)+0.20*control+0.20*boundary-0.20*dr
        bat_stats.append({"Player":player,"bat_opp_points":sc,"bat_opp_balls":balls})
    bowl_stats=[]
    for player in info["Player"]:
        pdf=dw[dw["bowl"]==player]; balls=len(pdf)
        if balls<min_balls: continue
        runs=np.nansum(pdf["batruns"]); wkts=np.nansum(pdf["out"])
        boundary=(pdf["batruns"].isin([4,6])).sum()/balls
        dots=(pdf["score"]==0).sum()/balls if "score" in pdf.columns else 0
        control=np.nanmean(pdf["control"]) if "control" in pdf.columns else 0
        rpb=runs/balls; wr=wkts/balls
        sc=0.45*(1-min(rpb/6,1))+0.25*wr+0.20*dots+0.10*(1-boundary)
        bowl_stats.append({"Player":player,"bowl_opp_points":sc,"bowl_opp_balls":balls})
    bf=pd.DataFrame(bat_stats); wf=pd.DataFrame(bowl_stats)
    info=info.merge(bf,on="Player",how="left").merge(wf,on="Player",how="left")
    info["bat_opp_points"]=info["bat_opp_points"].fillna(0)
    info["bowl_opp_points"]=info["bowl_opp_points"].fillna(0)
    info["bat_share"]=pd.to_numeric(info.get("bat_share",0.5),errors="coerce").fillna(0.5)
    info["bowl_share"]=pd.to_numeric(info.get("bowl_share",0.5),errors="coerce").fillna(0.5)
    info["opp_points_raw"]=info["bat_share"]*info["bat_opp_points"]+info["bowl_share"]*info["bowl_opp_points"]
    info["opp_percentile"]=info["opp_points_raw"].rank(pct=True)*100
    info["opp_adjust"]=alpha*np.tanh((info["opp_percentile"]-50)/25)
    info["adjusted_prob_before_opp"]=info["adjusted_prob"]
    info["adjusted_prob"]=np.clip(info["adjusted_prob"]+info["opp_adjust"],0,100)
    info=info.sort_values("adjusted_prob",ascending=False).reset_index(drop=True)
    info["adjusted_rank"]=np.arange(1,len(info)+1)
    od=info[["Team","Player","Role","bat_opp_points","bowl_opp_points","opp_points_raw","opp_percentile","opp_adjust","adjusted_prob_before_opp","adjusted_prob","adjusted_rank"]].copy()
    return info, od

def apply_bowl_style_venue_adjustment(team_adjusted, ipl_df, target_venue, alpha=1.3, print_details=False):
    info=team_adjusted.copy(); df=ipl_df.copy()
    df["ground"]=df["ground"].astype(str).str.lower().str.strip()
    df["bowl_style"]=df["bowl_style"].astype(str).str.upper().str.strip()
    info["Player"]=info["Player"].astype(str).str.lower().str.strip()
    info["bowl_style"]=info["bowl_style"].astype(str).str.upper().str.strip()
    tv=str(target_venue).lower().strip()
    info["adjusted_prob"]=pd.to_numeric(info["adjusted_prob"],errors="coerce").fillna(0.0)
    info["adjusted_prob_before_bowl_style"]=info["adjusted_prob"]
    vdf=df[df["ground"]==tv].copy()
    if vdf.empty:
        info["bowl_style_adjust"]=0.0
        return info, pd.DataFrame(), pd.DataFrame()
    styles=vdf["bowl_style"].dropna().unique().tolist()
    rows=[]
    for style in styles:
        sg=vdf[vdf["bowl_style"]==style].copy()
        balls=len(sg)
        if balls<10: continue
        rc=float(sg["batruns"].sum()) if "batruns" in sg else 0.0
        wk=float(sg["out"].sum()) if "out" in sg else 0.0
        dots=float((sg["score"]==0).sum()) if "score" in sg else 0.0
        bnd=float(sg["batruns"].isin([4,6]).sum()) if "batruns" in sg else 0.0
        ctrl=float(sg["control"].mean()) if "control" in sg.columns and sg["control"].notna().any() else 0.0
        rpb=rc/balls; wr=wk/balls; dr=dots/balls; br=bnd/balls
        ss=0.45*(1-min(rpb/6,1))+0.25*min(wr,1)+0.20*dr+0.10*(1-br)
        rows.append({"bowl_style":style,"balls":balls,"runs_conceded":rc,"wickets":wk,"dots":dots,"boundaries":bnd,"control_rate":ctrl,"runs_per_ball":rpb,"wicket_rate":wr,"dot_rate":dr,"boundary_rate":br,"style_score":ss})
    style_stats=pd.DataFrame(rows)
    if style_stats.empty:
        info["bowl_style_adjust"]=0.0
        return info, style_stats, pd.DataFrame()
    style_stats["total_percentile"]=style_stats["style_score"].rank(pct=True)*100
    style_stats["bowl_style_adjust"]=alpha*np.tanh((style_stats["total_percentile"]-50)/25)
    sam=style_stats.set_index("bowl_style")["bowl_style_adjust"]
    info["bowl_style_adjust"]=info["bowl_style"].map(sam).fillna(0.0)
    info["adjusted_prob"]=np.clip(info["adjusted_prob"]+info["bowl_style_adjust"],0,100)
    psd=info[["Team","Player","Role","bowl_style","adjusted_prob_before_bowl_style","bowl_style_adjust","adjusted_prob"]].copy()
    return info, style_stats, psd

def assign_final_batting_positions(df):
    def compute_positions(group):
        fa=group.groupby("bat")["ball_id"].min().sort_values()
        positions={bat:i+1 for i,bat in enumerate(fa.index)}
        group["batting_position"]=group["bat"].map(positions)
        return group
    df=df.groupby(["p_match","inns"],group_keys=False).apply(compute_positions)
    def get_mode(series):
        mv=series.mode()
        return int(mv.min()) if not mv.empty else None
    fp=df.groupby("bat")["batting_position"].agg(get_mode).reset_index().rename(columns={"batting_position":"final_batting_position"})
    return df.merge(fp,on="bat",how="left")

def apply_bowling_phase_adjustment(team_adjusted, mega_df, alpha=1.3, beta=25.0, min_balls=12, print_details=False):
    info=team_adjusted.copy(); data=mega_df.copy()
    info["Player"]=info["Player"].astype(str).str.lower().str.strip()
    info["Role"]=info["Role"].astype(str).str.lower().str.strip()
    info["adjusted_prob"]=pd.to_numeric(info["adjusted_prob"],errors="coerce").fillna(0.0)
    info["adjusted_prob_before_phase"]=info["adjusted_prob"]
    data["bowl"]=data["bowl"].astype(str).str.lower().str.strip()
    data["phase"]=data["phase"].astype(str).str.lower().str.strip()
    for c in ["bowlruns","ballfaced","out","score","noball","wide","batruns"]:
        data[c]=pd.to_numeric(data.get(c,np.nan),errors="coerce").fillna(0)
    data["legal_ball"]=((data["noball"]==0)&(data["wide"]==0)).astype(int)
    eligible_roles={"bowler","allrounder","bowling allrounder","batting allrounder"}
    def parse_bowl_role(val):
        if pd.isna(val): return []
        s=str(val).strip()
        if s in {"","nan","none","None"}: return []
        try:
            raw=ast.literal_eval(s)
            if not isinstance(raw,(list,tuple,set)): raw=[raw]
        except:
            raw=[x.strip().strip("'\"") for x in s.strip("[](){}").split(",") if x.strip()]
        phases=set()
        for item in raw:
            t=str(item).strip().lower()
            if "middle" in t: phases.add("middle 1"); phases.add("middle 2")
            elif "powerplay" in t: phases.add("powerplay")
            elif "death" in t: phases.add("death")
        return sorted(phases)
    def bowling_score(g):
        b=float(g["legal_ball"].sum())
        if b<=0: return 0.0
        rc=float(g["bowlruns"].sum()); wk=float(g["out"].sum())
        rpb=rc/b; wr=wk/b
        dr=float((g["score"]==0).sum())/len(g) if len(g)>0 else 0.0
        br=float(g["batruns"].isin([4,6]).sum())/len(g) if len(g)>0 else 0.0
        return 0.45*(1-min(rpb/6,1))+0.25*min(wr,1)+0.20*dr+0.10*(1-br)
    rows=[]
    for _,row in info.iterrows():
        player=row["Player"]; role=row["Role"]; br_val=row.get("bowl_role",None)
        ps=0.0; pb=0; pm=0; ap=parse_bowl_role(br_val)
        if role in eligible_roles and len(ap)>0:
            pdf=data[(data["bowl"]==player)&(data["phase"].isin(ap))].copy()
            if not pdf.empty:
                pb=int(pdf["legal_ball"].sum()); pm=int(pdf["p_match"].nunique())
                if pb>=min_balls: ps=bowling_score(pdf)
        rows.append({"Player":player,"phase_score":ps,"phase_balls":pb,"phase_matches":pm,"allowed_phases":ap})
    sdf=pd.DataFrame(rows)
    valid=sdf[sdf["phase_balls"]>=min_balls].copy()
    if not valid.empty:
        valid["phase_percentile"]=valid["phase_score"].rank(pct=True)*100.0
        valid["phase_adjust"]=alpha*np.tanh((valid["phase_percentile"]-50.0)/beta)
        am=valid.set_index("Player")["phase_adjust"]; pm2=valid.set_index("Player")["phase_percentile"]
        sdf["phase_percentile"]=sdf["Player"].map(pm2); sdf["phase_adjust"]=sdf["Player"].map(am).fillna(0.0)
    else:
        sdf["phase_percentile"]=np.nan; sdf["phase_adjust"]=0.0
    info=info.merge(sdf[["Player","allowed_phases","phase_matches","phase_balls","phase_score","phase_percentile","phase_adjust"]],on="Player",how="left")
    for c,f in [("phase_matches",0),("phase_balls",0),("phase_score",0.0),("phase_adjust",0.0)]:
        info[c]=info[c].fillna(f)
    info["phase_matches"]=info["phase_matches"].astype(int); info["phase_balls"]=info["phase_balls"].astype(int)
    info["phase_percentile"]=info["phase_percentile"].fillna(np.nan)
    info["adjusted_prob_before_phase"]=info["adjusted_prob"]
    info["adjusted_prob"]=np.clip(info["adjusted_prob"]+info["phase_adjust"],0,100)
    info=info.sort_values("adjusted_prob",ascending=False).reset_index(drop=True)
    info["adjusted_rank"]=np.arange(1,len(info)+1)
    pd_=info[["Team","Player","Role","bowl_role","allowed_phases","phase_matches","phase_balls","phase_score","phase_percentile","adjusted_prob_before_phase","phase_adjust","adjusted_prob","adjusted_rank"]].copy()
    return info, sdf, pd_

def can_play_position(row, pos):
    role=str(row["Role"]).strip().lower() if pd.notna(row["Role"]) else ""
    role2=str(row["bat_role_2"]).strip().lower() if pd.notna(row.get("bat_role_2","")) else ""
    limits=ROLE_POS_LIMITS.get(role)
    if limits:
        mn,mx=limits
        if not(mn<=pos<=mx): return False
    if role in ("bowler","bowling allrounder"):
        return pos==8 and role2 in ["lower order","other"]
    if pos in [1,2]: return role2 in ["opener"]
    elif pos==3: return role2 in ["opener","top order"]
    elif pos==4: return role2 in ["top order","middle order"]
    elif pos in [5,6,7]: return role2 in ["middle order","lower order"]
    elif pos==8: return role2 in ["lower order","other"]
    return False

def generate_batting_configs(fragment_name, positions, df):
    compatible_idx=[i for i,row in df.iterrows() if any(can_play_position(row,p) for p in positions)]
    configs=[]
    for comb in itertools.combinations(compatible_idx,len(positions)):
        for perm in itertools.permutations(comb):
            assignment=dict(zip(positions,perm))
            if not all(can_play_position(df.loc[p_idx],pos) for pos,p_idx in assignment.items()): continue
            players_set=set(comb)
            overseas_cnt=int(df.loc[list(players_set),"overseas"].sum())
            if overseas_cnt>2: continue
            bat_contrib=sum(BAT_SHARE_MAP[pos]*df.loc[p_idx,"adjusted_prob"] for pos,p_idx in assignment.items())
            wk_cnt=int((df.loc[list(players_set),"Role"]=="wicketkeeper batter").sum())
            bowl_cnt=int((df.loc[list(players_set),"bowl_share"]>0).sum())
            configs.append({"fragment":fragment_name,"players":players_set,"assignment":assignment,"bat_contrib":round(bat_contrib,4),"overseas":overseas_cnt,"wk":wk_cnt,"bowlers":bowl_cnt,"player_names":[df.loc[i,"Player"] for i in sorted(players_set)]})
    return configs

def generate_bowling_configs(fragment_name, n_players, style_filter, df, must_include_idx=None):
    candidates=[i for i,row in df.iterrows() if row["bowl_share"]>0 and style_filter(row)]
    if must_include_idx is not None and must_include_idx not in candidates:
        raise ValueError(f"Forced player idx={must_include_idx} not eligible for {fragment_name}")
    configs=[]
    for comb in itertools.combinations(candidates,n_players):
        if must_include_idx is not None and must_include_idx not in comb: continue
        players_set=set(comb)
        configs.append({"fragment":fragment_name,"players":players_set,"assignment":None,"bat_contrib":0.0,"overseas":int(df.loc[list(players_set),"overseas"].sum()),"wk":int((df.loc[list(players_set),"Role"]=="wicketkeeper batter").sum()),"bowlers":len(players_set),"player_names":[df.loc[i,"Player"] for i in sorted(players_set)]})
    return configs

def run_playing_xi_optimizer(team_adjusted_in, mega_df_in):
    df=team_adjusted_in.reset_index(drop=True).copy()
    df=df[df["adjusted_prob"]>0].reset_index(drop=True)
    for role,share in ROLE_TO_BOWL_SHARE.items():
        df.loc[df["Role"]==role,"bowl_share"]=share
    pace_candidates=df[df["bowl_style"].isin(PACE_STYLES)&df["Role"].isin(["bowler","bowling allrounder"])].copy()
    if len(pace_candidates)==0:
        return None, "No pace bowler/bowling-allrounder found."
    bpr=pace_candidates.loc[pace_candidates["adjusted_prob"].idxmax()]
    best_pacer=bpr["Player"]; best_pacer_idx=bpr.name
    f1=generate_batting_configs("F1",[1,2],df)
    f2=generate_batting_configs("F2",[3,4,5],df)
    f3=generate_batting_configs("F3",[6,7,8],df)
    f4=generate_bowling_configs("F4",1,lambda r:pd.notna(r["bowl_style"]) and str(r["bowl_style"]).strip() in SPIN_STYLES,df)
    try:
        f5=generate_bowling_configs("F5",2,lambda r:pd.notna(r["bowl_style"]) and str(r["bowl_style"]).strip() in PACE_STYLES,df,must_include_idx=best_pacer_idx)
    except ValueError as e:
        return None, str(e)
    f6=generate_bowling_configs("F6",2,lambda r:True,df)
    valid_teams=[]
    for c1 in f1:
        for c2 in f2:
            for c3 in f3:
                batting_players=c1["players"]|c2["players"]|c3["players"]
                if len(batting_players)!=8: continue
                bat_total=c1["bat_contrib"]+c2["bat_contrib"]+c3["bat_contrib"]
                pos_1_to_7=c1["players"]|c2["players"]
                for pos,pidx in c3["assignment"].items():
                    if pos in [6,7]: pos_1_to_7.add(pidx)
                for c4 in f4:
                    if c4["players"]&pos_1_to_7: continue
                    for c5 in f5:
                        if c5["players"]&pos_1_to_7: continue
                        for c6 in f6:
                            bowling_players=c4["players"]|c5["players"]|c6["players"]
                            if len(bowling_players)!=5: continue
                            total_players=batting_players|bowling_players
                            if len(total_players)>12: continue
                            overlap_set=batting_players&bowling_players
                            if len(overlap_set)>0 and not overlap_set.issubset(c6["players"]): continue
                            pl=list(total_players)
                            to=int(df.loc[pl,"overseas"].sum()); tw=int((df.loc[pl,"Role"]=="wicketkeeper batter").sum()); tb=int((df.loc[pl,"bowl_share"]>0).sum())
                            if to>4 or tw<1 or tb<6: continue
                            bowl_total=sum(df.loc[idx,"bowl_share"]*df.loc[idx,"adjusted_prob"] for idx in total_players)
                            ter=round(bat_total+bowl_total,4)
                            batting_order=["---"]*9
                            for c in [c1,c2,c3]:
                                for pos,pidx in c["assignment"].items(): batting_order[pos]=df.loc[pidx,"Player"]
                            overlap_name=df.loc[list(overlap_set)[0],"Player"] if overlap_set else "None"
                            pf={}
                            for pidx in c1["players"]: pf[df.loc[pidx,"Player"]]="F1"
                            for pidx in c2["players"]: pf[df.loc[pidx,"Player"]]="F2"
                            for pidx in c3["players"]: pf[df.loc[pidx,"Player"]]="F3"
                            for pidx in c4["players"]: pf[df.loc[pidx,"Player"]]="F4"
                            for pidx in c5["players"]: pf[df.loc[pidx,"Player"]]="F5"
                            for pidx in c6["players"]: pf[df.loc[pidx,"Player"]]="F6"
                            if overlap_name!="None": pf[overlap_name]="F6"
                            valid_teams.append({"TER":ter,"batting_order_1_to_8":batting_order[1:9],"overlap_AR":overlap_name,"all_players":sorted([df.loc[i,"Player"] for i in total_players]),"total_overseas":to,"total_WK":tw,"total_bowlers_capable":tb,"F1":c1["player_names"],"F2":c2["player_names"],"F3":c3["player_names"],"F4_spin":c4["player_names"],"F5_pace":c5["player_names"],"F6_any":c6["player_names"],"player_fragment":pf})
    if not valid_teams: return None, "No valid team found satisfying all constraints."
    valid_teams.sort(key=lambda x:x["TER"],reverse=True)
    best=valid_teams[0]
    remaining=[p for p in best["all_players"] if p not in best["batting_order_1_to_8"][:7]]
    ars=[p for p in remaining if "allrounder" in df.loc[df["Player"]==p,"Role"].iloc[0].lower()]
    bwls=[p for p in remaining if "allrounder" not in df.loc[df["Player"]==p,"Role"].iloc[0].lower()]
    ars.sort(key=lambda p:df.loc[df["Player"]==p,"adjusted_prob"].iloc[0],reverse=True)
    def bwt(pname):
        try:
            d=mega_df_in[mega_df_in["bat"]==pname] if "bat" in mega_df_in.columns else pd.DataFrame()
            if d.empty: d=mega_df_in[mega_df_in["Player"]==pname] if "Player" in mega_df_in.columns else pd.DataFrame()
            r=d["batruns"].sum() if (len(d)>0 and "batruns" in d.columns) else 0
            b=d["ballfaced"].sum() if (len(d)>0 and "ballfaced" in d.columns) else 0
            sr=(r/b*100) if b>0 else 0.0
            return 0.5*r+0.5*sr
        except: return 0.0
    bwls.sort(key=bwt,reverse=True)
    order_8_12=ars+bwls
    rows=[]
    for pos in range(1,8):
        pname=best["batting_order_1_to_8"][pos-1]
        idx=df[df["Player"]==pname].index[0]; row=df.loc[idx]
        rows.append({"Pos":pos,"Player":pname.title(),"Role":row["Role"],"Fragment":best["player_fragment"].get(pname,"?"),"adj_prob":round(row["adjusted_prob"],4),"Overseas":int(row["overseas"])})
    for pos,pname in enumerate(order_8_12,start=8):
        idx=df[df["Player"]==pname].index[0]; row=df.loc[idx]
        rows.append({"Pos":pos,"Player":pname.title(),"Role":row["Role"],"Fragment":best["player_fragment"].get(pname,"?"),"adj_prob":round(row["adjusted_prob"],4),"Overseas":int(row["overseas"])})
    result_df=pd.DataFrame(rows)
    return {"team_df":result_df,"best":best,"best_pacer":best_pacer,"valid_count":len(valid_teams)}, None

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_base_data(data_dir):
    parquet_files=glob.glob(os.path.join(data_dir,"*.parquet"))
    if not parquet_files: raise FileNotFoundError("No .parquet files found in data directory.")
    df_raw=pd.concat([pd.read_parquet(f) for f in parquet_files],ignore_index=True)
    df_raw=df_raw[df_raw["year"]>=2024]
    teams_df=pd.read_csv(os.path.join(data_dir,"IPL_teams_2026.csv"))
    icr_pred=pd.read_csv(os.path.join(data_dir,"icr_2026_final.csv"))
    bicr_pred=pd.read_csv(os.path.join(data_dir,"bicr_2026 (1).csv"))
    icr_2025=pd.read_csv(os.path.join(data_dir,"icr_2025.csv"))
    bicr_2025=pd.read_csv(os.path.join(data_dir,"bicr_2025.csv"))
    form_df=pd.read_csv(os.path.join(data_dir,"form_data.csv"))
    ipl_df=pd.read_csv(os.path.join(data_dir,"ipl_bbb_2021_25.csv"))
    ipl_df=ipl_df[ipl_df["year"]>2022]
    return df_raw, teams_df, icr_pred, bicr_pred, icr_2025, bicr_2025, form_df, ipl_df

def build_venue_df():
    vdf=pd.read_csv(StringIO(VENUE_TEXT),sep="\t")
    vdf.rename(columns={"Unnamed: 0":"Unnamed"},inplace=True)
    vdf["venue"]=vdf["venue"].astype(str).str.strip()
    vdf["venue_factor"]=pd.to_numeric(vdf["venue_factor"],errors="coerce")
    vdf["spin_index"]=pd.to_numeric(vdf["spin_index"],errors="coerce")
    vdf=vdf.dropna(subset=["venue","venue_factor","spin_index"])
    return vdf.groupby("venue",as_index=False).agg(venue_factor=("venue_factor","mean"),spin_index=("spin_index","mean"))

def build_result_df(df_raw, teams_df, icr_pred, bicr_pred, players_to_remove):
    icr_pred=icr_pred.rename(columns={"icr_pred":"icr_pred_1","icr_percentile_final_norm":"icr_pred"})
    bicr_pred=bicr_pred.rename(columns={"icr_pred":"icr_pred_1","icr_percentile_final_norm":"icr_pred"})
    bicr_pred["icr_pred"]=bicr_pred["player"].map(PREDEFINED_ICR).fillna(bicr_pred["icr_pred"])
    icr_pred["icr_pred"]=icr_pred["player"].map(PREDEFINED_ICR).fillna(icr_pred["icr_pred"])
    teams=teams_df.copy()
    teams["Player"]=teams["Player"].astype(str).str.lower().str.strip()
    icr_pred["player"]=icr_pred["player"].astype(str).str.lower().str.strip()
    bicr_pred["player"]=bicr_pred["player"].astype(str).str.lower().str.strip()
    for col in ["ICR_pred","BICR_pred","bat_matches","bowl_matches","total_matches","bat_share","bowl_share","baseline_raw","team_total","baseline_prob"]:
        if col in teams.columns: teams=teams.drop(columns=[col])
    idf=icr_pred[["player","icr_pred"]].rename(columns={"player":"Player","icr_pred":"ICR_pred"})
    bdf=bicr_pred[["player","icr_pred"]].rename(columns={"player":"Player","icr_pred":"BICR_pred"})
    teams=teams.merge(idf,on="Player",how="left").merge(bdf,on="Player",how="left")
    balls=df_raw.copy()
    balls["bat"]=balls["bat"].astype(str).str.lower().str.strip()
    balls["bowl"]=balls["bowl"].astype(str).str.lower().str.strip()
    bp=balls.loc[balls["bat"].notna()&(balls["bat"]!="nan"),["p_match","bat"]].drop_duplicates().rename(columns={"bat":"Player"})
    wp=balls.loc[balls["bowl"].notna()&(balls["bowl"]!="nan"),["p_match","bowl"]].drop_duplicates().rename(columns={"bowl":"Player"})
    bmc=bp.groupby("Player")["p_match"].nunique().to_dict()
    wmc=wp.groupby("Player")["p_match"].nunique().to_dict()
    ap=pd.concat([bp[["Player","p_match"]],wp[["Player","p_match"]]],ignore_index=True).drop_duplicates()
    tmc=ap.groupby("Player")["p_match"].nunique().to_dict()
    teams["bat_matches"]=teams["Player"].map(bmc).fillna(0)
    teams["bowl_matches"]=teams["Player"].map(wmc).fillna(0)
    teams["total_matches"]=teams["Player"].map(tmc).fillna(0)
    teams["bat_share"]=np.where(teams["total_matches"]>0,teams["bat_matches"]/teams["total_matches"],np.nan)
    teams["bowl_share"]=np.where(teams["total_matches"]>0,teams["bowl_matches"]/teams["total_matches"],np.nan)
    teams["Role_clean"]=teams["Role"].astype(str).str.lower().str.strip()
    bat_roles={"batter","middle order batter","wicketkeeper batter","top order batter","opening batter"}
    mask_no=teams["total_matches"]==0
    teams.loc[mask_no&teams["Role_clean"].isin(bat_roles),"bat_share"]=1.0
    teams.loc[mask_no&teams["Role_clean"].isin(bat_roles),"bowl_share"]=0.0
    teams.loc[mask_no&(teams["Role_clean"]=="bowler"),"bat_share"]=0.0
    teams.loc[mask_no&(teams["Role_clean"]=="bowler"),"bowl_share"]=1.0
    teams.loc[mask_no&(teams["Role_clean"]=="allrounder"),"bat_share"]=0.5
    teams.loc[mask_no&(teams["Role_clean"]=="allrounder"),"bowl_share"]=0.5
    teams.loc[mask_no&(teams["Role_clean"]=="bowling allrounder"),"bat_share"]=0.4
    teams.loc[mask_no&(teams["Role_clean"]=="bowling allrounder"),"bowl_share"]=0.6
    teams.loc[mask_no&(teams["Role_clean"]=="batting allrounder"),"bat_share"]=0.6
    teams.loc[mask_no&(teams["Role_clean"]=="batting allrounder"),"bowl_share"]=0.4
    teams["bat_share"]=teams["bat_share"].fillna(0.5)
    teams["bowl_share"]=teams["bowl_share"].fillna(0.5)
    teams["baseline_raw"]=teams["bat_share"]*teams["ICR_pred"].fillna(0)+teams["bowl_share"]*teams["BICR_pred"].fillna(0)
    teams["team_total_icr_bicr"]=teams.groupby("Team").apply(lambda x:(x["ICR_pred"].fillna(0)+x["BICR_pred"].fillna(0)).sum()).reindex(teams["Team"]).to_numpy()
    teams["baseline_prob"]=np.where(teams["team_total_icr_bicr"]>0,teams["baseline_raw"]/teams["team_total_icr_bicr"],0)
    result=teams[["Team","Player","Role","bat_matches","bowl_matches","total_matches","bat_share","bowl_share","ICR_pred","BICR_pred","baseline_raw","team_total_icr_bicr","baseline_prob"]]
    result=result[~result["Player"].isin(players_to_remove)]
    return result

# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
for key,val in [("data_loaded",False),("pipeline_done",False),("team_adjusted",None),
                ("mega_df",None),("venue_df",None),("similar_venue_dataset",None),
                ("result",None),("ipl_df",None),("icr_2025",None),("bicr_2025",None),
                ("form_detail",None),("venue_detail",None),("matchup_detail",None),
                ("opp_detail",None),("bowl_style_detail",None),("phase_detail",None),
                ("playing_xi_result",None),("our_team",""),("opposition",""),("venue","")]:
    if key not in st.session_state: st.session_state[key]=val

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🏏 IPL Decision Intelligence Engine</h1>
  <div class="subtitle">Data-driven squad selection & match strategy platform for IPL coaches</div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data Configuration")
    data_dir = st.text_input("Data Directory", value=".", help="Path containing .parquet files and CSVs")
    players_to_remove_input = st.text_area(
        "Players to Exclude (one per line)",
        value="\n".join(DEFAULT_PLAYERS_TO_REMOVE),
        height=120
    )
    players_to_remove = [p.strip().lower() for p in players_to_remove_input.strip().split("\n") if p.strip()]

    if st.button("📂 Load Data Files", use_container_width=True):
        with st.spinner("Loading data files..."):
            try:
                df_raw, teams_df_raw, icr_pred, bicr_pred, icr_2025, bicr_2025, form_df_raw, ipl_df_raw = load_base_data(data_dir)
                result = build_result_df(df_raw, teams_df_raw, icr_pred, bicr_pred, players_to_remove)
                venue_df = build_venue_df()
                mega = df_raw.copy()
                mega["bat"] = mega["bat"].astype(str).str.lower().str.strip()
                mega["bowl"] = mega["bowl"].astype(str).str.lower().str.strip()
                mega["ground"] = mega["ground"].astype(str).str.strip()
                mega = assign_final_batting_positions(mega)
                mega = mega.merge(venue_df[["venue","venue_factor","spin_index"]], left_on="ground", right_on="venue", how="left")
                scols = ["p_match","inns","ball_id","team_bat","team_bowl","bat","bowl"]
                mega = mega.drop_duplicates(subset=[c for c in scols if c in mega.columns])
                if "over" in mega.columns:
                    def map_phase(o):
                        if pd.isna(o): return "unknown"
                        o=int(o)
                        if o<=6: return "Powerplay"
                        elif o<=11: return "Middle 1"
                        elif o<=16: return "Middle 2"
                        return "Death"
                    mega["phase"] = mega["over"].apply(map_phase)
                st.session_state.data_loaded = True
                st.session_state.result = result
                st.session_state.mega_df = mega
                st.session_state.venue_df = venue_df
                st.session_state.ipl_df = ipl_df_raw
                st.session_state.icr_2025 = icr_2025
                st.session_state.bicr_2025 = bicr_2025
                st.session_state.form_df = form_df_raw
                st.session_state.teams_df = teams_df_raw
                st.success(f"✅ Data loaded — {len(df_raw):,} balls, {result['Team'].nunique()} teams")
            except Exception as e:
                st.error(f"❌ {e}")

    st.markdown("---")

    if st.session_state.data_loaded:
        st.markdown("### ⚙️ Match Configuration")
        result = st.session_state.result
        teams_list = sorted(result["Team"].unique().tolist())
        venue_list = sorted(st.session_state.venue_df["venue"].tolist())

        our_team = st.selectbox("🏏 Our Team", teams_list, key="our_team_select")
        opp_options = [t for t in teams_list if t != our_team]
        opposition = st.selectbox("🆚 Opposition", opp_options, key="opp_select")
        venue = st.selectbox("🏟️ Venue", venue_list, key="venue_select")

        st.markdown("**Overseas Players (Our Team)**")
        our_squad = sorted(result[result["Team"]==our_team]["Player"].str.title().tolist())
        overseas_selected = st.multiselect("Mark overseas players", our_squad, key="overseas_select")
        overseas_set = set(p.lower() for p in overseas_selected)

        st.markdown("**Opposition XI**")
        opp_squad = sorted(result[result["Team"]==opposition]["Player"].str.title().tolist())
        opp_xi_selected = st.multiselect("Select opposition playing XI (11 players)", opp_squad, key="opp_xi_select", max_selections=12)
        opposition_xi = {i+1: p for i, p in enumerate(opp_xi_selected)}

        st.markdown("---")
        st.markdown("### 🎚️ Adjustment Parameters")
        form_alpha = 2.75
        venue_alpha = st.slider("Venue α", 0.5, 3.0, 1.5, 0.25)
        matchup_alpha = st.slider("Matchup α", 0.5, 3.0, 1.25, 0.25)
        opp_alpha = st.slider("Opposition α", 0.5, 3.0, 1.0, 0.25)
        bowl_alpha = st.slider("Bowl Style α", 0.5, 3.0, 1.3, 0.25)
        phase_alpha = st.slider("Phase α", 0.5, 3.0, 1.3, 0.25)

        st.markdown("---")
        run_btn = st.button("🚀 Run Full Analysis", use_container_width=True, type="primary")

        if run_btn:
            if not our_team or not opposition or not venue:
                st.error("Select team, opposition and venue first.")
            elif len(opp_xi_selected) < 11:
                st.warning("Select at least 11 opposition XI players.")
            else:
                progress = st.progress(0, text="Starting pipeline...")
                try:
                    # Step 1: Form
                    progress.progress(10, "Running Form Analysis...")
                    ta, pmd, ctx = build_team_form_adjusted_prob(our_team, opposition, venue, st.session_state.result, st.session_state.form_df)
                    ta["Player"] = ta["Player"].astype(str).str.lower().str.strip()
                    ta["overseas"] = ta["Player"].isin(overseas_set).astype(int)

                    # Merge bowl style
                    mega = st.session_state.mega_df
                    bl = mega[["bowl","bowl_style"]].dropna().drop_duplicates(subset=["bowl"]).rename(columns={"bowl":"player_lower"})
                    ta["player_lower"] = ta["Player"]
                    ta = ta.merge(bl, on="player_lower", how="left").drop(columns=["player_lower"])

                    # Merge bat/bowl roles from 2025 csvs
                    ic25 = st.session_state.icr_2025.copy(); bc25 = st.session_state.bicr_2025.copy()
                    ic25["player_lower"] = ic25["player"].astype(str).str.lower()
                    bc25["player_lower"] = bc25["player"].astype(str).str.lower()
                    ta["player_lower"] = ta["Player"]
                    ta = ta.merge(ic25[["player_lower","comp_group"]], on="player_lower", how="left").rename(columns={"comp_group":"bat_role"})
                    ta = ta.merge(bc25[["player_lower","comp_group"]], on="player_lower", how="left").rename(columns={"comp_group":"bowl_role"})
                    ta = ta.drop(columns=["player_lower"])

                    # bat_pos and bat_role_2
                    mega["bowl_lower"] = mega["bat"].astype(str).str.lower()
                    bpl = mega.groupby("bowl_lower")["final_batting_position"].mean().reset_index()
                    bpl["bat_pos"] = np.floor(bpl["final_batting_position"]).fillna(8).astype(int)
                    ta["player_lower"] = ta["Player"]
                    ta = ta.merge(bpl[["bowl_lower","bat_pos"]], left_on="player_lower", right_on="bowl_lower", how="left").drop(columns=["player_lower","bowl_lower"])
                    ta["bat_pos"] = ta["bat_pos"].fillna(8).astype(int)
                    def classify_bat_role(pos):
                        if pos<3: return "Opener"
                        elif pos in [3,4]: return "Top Order"
                        elif pos in [5,6]: return "Middle Order"
                        elif pos in [7,8]: return "Lower Order"
                        return "Other"
                    ta["bat_role_2"] = ta["bat_pos"].apply(classify_bat_role)

                    st.session_state.form_detail = pmd
                    progress.progress(25, "Running Venue Analysis...")

                    # Step 2: Similar venues
                    svd = build_similar_venue_dataset(venue, mega, st.session_state.venue_df)
                    st.session_state.similar_venue_dataset = svd

                    # Step 3: Venue adjustment
                    ta, vst, vd, _ = apply_venue_adjustment_role_weighted(ta, mega, svd, venue, alpha=venue_alpha, beta=25.0, min_matches=3)
                    st.session_state.venue_detail = vd
                    progress.progress(45, "Running Matchup Analysis...")

                    # Step 4: Matchup
                    ta, md, bpd, wpd = apply_matchup_adjustment(ta, mega, our_team, opposition_xi, alpha=matchup_alpha, min_balls=5)
                    st.session_state.matchup_detail = md
                    progress.progress(60, "Running Opposition Analysis...")

                    # Step 5: Opposition
                    ta, od = apply_opposition_adjustment(ta, st.session_state.ipl_df, our_team, opposition, alpha=opp_alpha, min_balls=40)
                    st.session_state.opp_detail = od
                    progress.progress(72, "Running Bowl Style Analysis...")

                    # Step 6: Bowl style
                    ta, bss, bsd = apply_bowl_style_venue_adjustment(ta, st.session_state.ipl_df, venue, alpha=bowl_alpha)
                    st.session_state.bowl_style_detail = bsd
                    progress.progress(84, "Running Phase Analysis...")

                    # Step 7: Phase
                    ta, phs, phd = apply_bowling_phase_adjustment(ta, mega, alpha=phase_alpha, beta=25.0, min_balls=12)
                    st.session_state.phase_detail = phd
                    progress.progress(95, "Optimising Playing XI...")

                    # Step 8: Playing XI
                    xi_result, err = run_playing_xi_optimizer(ta, mega)
                    if err: st.error(f"Optimizer: {err}")
                    else: st.session_state.playing_xi_result = xi_result

                    st.session_state.team_adjusted = ta
                    st.session_state.our_team = our_team
                    st.session_state.opposition = opposition
                    st.session_state.venue = venue
                    st.session_state.pipeline_done = True
                    progress.progress(100, "✅ Pipeline complete!")
                    st.success("Analysis complete! View results in the tabs →")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    import traceback; st.code(traceback.format_exc())

# ─── MATCH BANNER ─────────────────────────────────────────────────────────────
if st.session_state.pipeline_done:
    st.markdown(f"""
    <div class="match-banner">
      <div class="team">{st.session_state.our_team}</div>
      <div class="vs">VS</div>
      <div class="team">{st.session_state.opposition}</div>
      <div style="flex:1"></div>
      <div class="venue">📍 {st.session_state.venue}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN TABS ────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6,t7 = st.tabs([
    "📊 Baseline Squad", "📈 Form Analysis", "🏟️ Venue Analysis",
    "⚔️ Matchup Analysis", "🎯 Opposition Analysis",
    "🌀 Bowl Style & Phase", "🏆 Playing XI"
])

# ── TAB 1: Baseline ──────────────────────────────────────────────────────────
with t1:
    if not st.session_state.data_loaded:
        st.info("Load data files first using the sidebar.")
    else:
        result = st.session_state.result
        teams_list = sorted(result["Team"].unique().tolist())
        sel_team = st.selectbox("Select team to view", teams_list, key="t1_team")
        team_data = result[result["Team"]==sel_team].sort_values("baseline_prob",ascending=False).reset_index(drop=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="val">{len(team_data)}</div><div class="lbl">Squad Size</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="val">{team_data["baseline_prob"].max():.3f}</div><div class="lbl">Highest Prob</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="val">{team_data["ICR_pred"].mean():.1f}</div><div class="lbl">Avg ICR</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="val">{team_data["BICR_pred"].mean():.1f}</div><div class="lbl">Avg BICR</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Squad Baseline Probabilities</div>', unsafe_allow_html=True)
        disp = team_data[["Player","Role","ICR_pred","BICR_pred","bat_matches","bowl_matches","baseline_prob"]].copy()
        disp.columns = ["Player","Role","ICR","BICR","Bat Matches","Bowl Matches","Baseline Prob"]
        disp["Player"] = disp["Player"].str.title()
        st.dataframe(disp.style.background_gradient(subset=["Baseline Prob"], cmap="YlOrRd"), use_container_width=True, height=450)

# ── TAB 2: Form ──────────────────────────────────────────────────────────────
with t2:
    if not st.session_state.pipeline_done:
        st.info("Run the full analysis pipeline from the sidebar.")
    else:
        ta = st.session_state.team_adjusted
        st.markdown('<div class="section-title">Form-Adjusted Player Rankings</div>', unsafe_allow_html=True)
        if "form_bonus" in ta.columns:
            cols = ["Player","Role","baseline_prob","form_matches","total_points","form_bonus","adjusted_prob"]
            cols = [c for c in cols if c in ta.columns]
            disp = ta[cols].copy(); disp["Player"] = disp["Player"].str.title()
            st.dataframe(disp.sort_values("adjusted_prob",ascending=False).style.background_gradient(subset=["adjusted_prob"],cmap="YlOrRd"), use_container_width=True)
        if st.session_state.form_detail is not None:
            with st.expander("📋 Match-by-match form detail"):
                fd = st.session_state.form_detail.copy()
                fd["Player"] = fd["Player"].str.title()
                st.dataframe(fd, use_container_width=True)

# ── TAB 3: Venue ─────────────────────────────────────────────────────────────
with t3:
    if not st.session_state.pipeline_done:
        st.info("Run the full analysis pipeline from the sidebar.")
    else:
        svd = st.session_state.similar_venue_dataset
        if svd is not None and not svd.empty:
            st.markdown('<div class="section-title">Similar Venues Used</div>', unsafe_allow_html=True)
            st.dataframe(svd[["venue","venue_factor","spin_index","matches_played"]] if "matches_played" in svd.columns else svd, use_container_width=True)
        if st.session_state.venue_detail is not None:
            st.markdown('<div class="section-title">Venue Performance Adjustment</div>', unsafe_allow_html=True)
            vd = st.session_state.venue_detail.copy(); vd["Player"] = vd["Player"].str.title()
            cols = ["Player","Role","venue_matches","bat_venue_points","bowl_venue_points","total_percentile","venue_adjust","adjusted_prob_before_venue","adjusted_prob"]
            cols = [c for c in cols if c in vd.columns]
            st.dataframe(vd[cols].sort_values("adjusted_prob",ascending=False).style.background_gradient(subset=["venue_adjust"],cmap="RdYlGn"), use_container_width=True)

# ── TAB 4: Matchup ────────────────────────────────────────────────────────────
with t4:
    if not st.session_state.pipeline_done:
        st.info("Run the full analysis pipeline from the sidebar.")
    else:
        if st.session_state.matchup_detail is not None:
            md = st.session_state.matchup_detail.copy(); md["Player"] = md["Player"].str.title()
            st.markdown('<div class="section-title">Head-to-Head Matchup Analysis</div>', unsafe_allow_html=True)
            cols = ["Player","Role","bat_matchup_points","bat_matchup_pairs","bowl_matchup_points","bowl_matchup_pairs","matchup_adjust","adjusted_prob"]
            cols = [c for c in cols if c in md.columns]
            st.dataframe(md[cols].sort_values("matchup_adjust",ascending=False).style.background_gradient(subset=["matchup_adjust"],cmap="RdYlGn"), use_container_width=True)

# ── TAB 5: Opposition ─────────────────────────────────────────────────────────
with t5:
    if not st.session_state.pipeline_done:
        st.info("Run the full analysis pipeline from the sidebar.")
    else:
        if st.session_state.opp_detail is not None and not st.session_state.opp_detail.empty:
            od = st.session_state.opp_detail.copy(); od["Player"] = od["Player"].str.title()
            st.markdown('<div class="section-title">Performance vs Opposition</div>', unsafe_allow_html=True)
            cols = ["Player","Role","bat_opp_points","bowl_opp_points","opp_percentile","opp_adjust","adjusted_prob"]
            cols = [c for c in cols if c in od.columns]
            st.dataframe(od[cols].sort_values("opp_adjust",ascending=False).style.background_gradient(subset=["opp_adjust"],cmap="RdYlGn"), use_container_width=True)
        else:
            st.info("No head-to-head data available between these teams.")

# ── TAB 6: Bowl Style & Phase ─────────────────────────────────────────────────
with t6:
    if not st.session_state.pipeline_done:
        st.info("Run the full analysis pipeline from the sidebar.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Bowl Style at Venue</div>', unsafe_allow_html=True)
            if st.session_state.bowl_style_detail is not None and not st.session_state.bowl_style_detail.empty:
                bsd = st.session_state.bowl_style_detail.copy(); bsd["Player"] = bsd["Player"].str.title()
                cols = [c for c in ["Player","Role","bowl_style","bowl_style_adjust","adjusted_prob"] if c in bsd.columns]
                st.dataframe(bsd[cols].sort_values("bowl_style_adjust",ascending=False).style.background_gradient(subset=["bowl_style_adjust"],cmap="RdYlGn"), use_container_width=True)
            else: st.info("No bowl style data.")
        with col2:
            st.markdown('<div class="section-title">Phase Bowling Analysis</div>', unsafe_allow_html=True)
            if st.session_state.phase_detail is not None:
                phd = st.session_state.phase_detail.copy(); phd["Player"] = phd["Player"].str.title()
                cols = [c for c in ["Player","Role","allowed_phases","phase_matches","phase_balls","phase_score","phase_adjust","adjusted_prob"] if c in phd.columns]
                st.dataframe(phd[cols].sort_values("phase_adjust",ascending=False).style.background_gradient(subset=["phase_adjust"],cmap="RdYlGn"), use_container_width=True)
            else: st.info("No phase data.")

# ── TAB 7: Playing XI ─────────────────────────────────────────────────────────
with t7:
    if not st.session_state.pipeline_done or st.session_state.playing_xi_result is None:
        st.info("Run the full analysis pipeline from the sidebar.")
    else:
        xi = st.session_state.playing_xi_result
        best = xi["best"]
        team_df = xi["team_df"]

        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-top:3px solid #f4c430;border-radius:10px;padding:20px;margin-bottom:1.5rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
              <div style="font-size:0.78rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;">Optimal Team — {st.session_state.our_team}</div>
              <div style="font-size:1.6rem;font-weight:700;color:#f4c430;margin-top:4px;">TER: {best["TER"]:.4f}</div>
            </div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;">
              <div style="text-align:center;background:#21262d;border-radius:8px;padding:10px 18px;">
                <div style="font-size:1.3rem;font-weight:700;color:#e6edf3;">{best["total_overseas"]}</div>
                <div style="font-size:0.72rem;color:#8b949e;">Overseas</div>
              </div>
              <div style="text-align:center;background:#21262d;border-radius:8px;padding:10px 18px;">
                <div style="font-size:1.3rem;font-weight:700;color:#e6edf3;">{best["total_WK"]}</div>
                <div style="font-size:0.72rem;color:#8b949e;">Wicketkeepers</div>
              </div>
              <div style="text-align:center;background:#21262d;border-radius:8px;padding:10px 18px;">
                <div style="font-size:1.3rem;font-weight:700;color:#e6edf3;">{best["total_bowlers_capable"]}</div>
                <div style="font-size:0.72rem;color:#8b949e;">Bowl-capable</div>
              </div>
              <div style="text-align:center;background:#21262d;border-radius:8px;padding:10px 18px;">
                <div style="font-size:1.3rem;font-weight:700;color:#f4c430;">{xi["valid_count"]}</div>
                <div style="font-size:0.72rem;color:#8b949e;">Valid Combos</div>
              </div>
            </div>
          </div>
          <div style="margin-top:12px;font-size:0.82rem;color:#8b949e;">
            🔒 Best pacer locked in F5: <strong style="color:#e6edf3;">{xi["best_pacer"].title()}</strong>
            &nbsp;|&nbsp; Overlap AR: <strong style="color:#e6edf3;">{best["overlap_AR"].title()}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Batting order table
        rows_html = ""
        for _, row in team_df.iterrows():
            pos = row["Pos"]
            badge_cls = f"rank-{pos}" if pos <= 3 else "rank-badge"
            rb = role_badge(row["Role"])
            ovs_icon = "🌐" if row["Overseas"] else ""
            adj = row["adj_prob"]
            bar = prob_bar(adj)
            rows_html += f"""
            <tr>
              <td><span class="rank-badge {badge_cls}">{pos}</span></td>
              <td style="font-weight:600">{row["Player"]} {ovs_icon}</td>
              <td>{rb}</td>
              <td><span style="background:#21262d;padding:2px 8px;border-radius:4px;font-size:0.78rem;color:#8b949e;">{row["Fragment"]}</span></td>
              <td style="font-weight:600;color:#f4c430">{adj:.3f}</td>
              <td style="min-width:100px">{bar}</td>
            </tr>"""

        st.markdown(f"""
        <table class="player-table">
          <thead><tr>
            <th>Pos</th><th>Player</th><th>Role</th>
            <th>Fragment</th><th>Adj Prob</th><th>Rating</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:2rem;">Fragment Composition</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        for col, label, key in [(fc1,"F1 – Openers","F1"),(fc2,"F2 – Top/Middle","F2"),(fc3,"F3 – Lower Order","F3")]:
            with col:
                st.markdown(f"**{label}**")
                for p in best[key]: st.markdown(f"- {p.title()}")
        fc4, fc5, fc6 = st.columns(3)
        for col, label, key in [(fc4,"F4 – Spinner","F4_spin"),(fc5,"F5 – Pacers","F5_pace"),(fc6,"F6 – Any / AR","F6_any")]:
            with col:
                st.markdown(f"**{label}**")
                for p in best[key]: st.markdown(f"- {p.title()}")

        st.markdown('<div class="section-title" style="margin-top:2rem;">All Players Ranked by Adjusted Probability</div>', unsafe_allow_html=True)
        ta_disp = st.session_state.team_adjusted.copy()
        ta_disp["Player"] = ta_disp["Player"].str.title()
        disp_cols = ["Player","Role","baseline_prob","adjusted_prob","adjusted_rank"]
        disp_cols = [c for c in disp_cols if c in ta_disp.columns]
        st.dataframe(ta_disp[disp_cols].sort_values("adjusted_prob",ascending=False).style.background_gradient(subset=["adjusted_prob"],cmap="YlOrRd"), use_container_width=True)
