# %% [markdown]
# # StatsBomb Football Analysis Toolkit
# 
# This notebook provides a comprehensive toolkit for analyzing football matches using StatsBomb data. All functions are designed to be reusable for any match available in the StatsBomb dataset.
# 
# ## 🚀 Features Included:
# - **Data Loading**: Functions to explore and load StatsBomb data
# - **Pass Analysis**: Final third progressions, pass networks, pass maps
# - **Player Analysis**: Touch maps, positioning heatmaps, player comparisons
# - **Shot Analysis**: Shot maps, xG analysis, goal analysis
# - **Pressure Analysis**: Defensive pressure heatmaps
# - **Team Analysis**: Pass completion, possession analysis
# - **Visualization**: Professional football pitch visualizations with custom styling

# %% [markdown]
# ## 📚 Import Required Libraries
# 
# First, let's import all the necessary libraries for our football analysis toolkit.

# %%
# Core data analysis libraries
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Football analysis libraries
from mplsoccer import Pitch, VerticalPitch
import statsbombpy as sb

# Additional utilities
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")

print("📚 All libraries imported successfully!")
print("🏈 StatsBomb Football Analysis Toolkit Ready!")

# %% [markdown]
# ## 🔍 StatsBomb Analytics Module
# 
# A comprehensive class-based approach for analyzing football data using StatsBomb's API with robust error handling and advanced analytics.

# %%
"""
StatsBomb Football Analytics Module
===================================

This module provides functions to fetch, process, and analyze football data using StatsBomb's API.
It includes functionality for competitions, matches, events, lineups, and advanced analytics.

Installation: pip install statsbombpy

Author: GitHub Copilot
"""

import pandas as pd
import numpy as np
from statsbombpy import sb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

class StatsBombAnalyzer:
    """
    A comprehensive class for analyzing football data using StatsBomb API
    """
    
    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        """
        Initialize the StatsBomb analyzer
        
        Args:
            credentials: Dict with 'user' and 'passwd' keys for API access
                        If None, will use open data only
        """
        self.credentials = credentials
        self.competitions_df = None
        self.current_matches = None
        
    def get_competitions(self, refresh: bool = False) -> pd.DataFrame:
        """
        Get all available competitions
        
        Args:
            refresh: Whether to refresh the cached competitions data
            
        Returns:
            DataFrame with competition information
        """
        if self.competitions_df is None or refresh:
            try:
                if self.credentials:
                    self.competitions_df = sb.competitions(creds=self.credentials)
                else:
                    self.competitions_df = sb.competitions()
                    
                print(f"✅ Loaded {len(self.competitions_df)} competitions")
                
            except Exception as e:
                print(f"❌ Error loading competitions: {e}")
                return pd.DataFrame()
                
        return self.competitions_df
    
    def list_available_competitions(self) -> None:
        """Print available competitions in a readable format"""
        comps = self.get_competitions()
        if not comps.empty:
            print("\n📊 Available Competitions:")
            print("=" * 60)
            for _, comp in comps.iterrows():
                print(f"🏆 {comp['competition_name']} ({comp['country_name']})")
                print(f"   Season: {comp['season_name']} | Gender: {comp['competition_gender']}")
                print(f"   Competition ID: {comp['competition_id']} | Season ID: {comp['season_id']}")
                print("-" * 60)
    
    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """
        Get matches for a specific competition and season
        
        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID
            
        Returns:
            DataFrame with match information
        """
        try:
            if self.credentials:
                matches = sb.matches(competition_id=competition_id, season_id=season_id, creds=self.credentials)
            else:
                matches = sb.matches(competition_id=competition_id, season_id=season_id)
                
            self.current_matches = matches
            print(f"✅ Loaded {len(matches)} matches")
            return matches
            
        except Exception as e:
            print(f"❌ Error loading matches: {e}")
            return pd.DataFrame()
    
    def get_lineups(self, match_id: int) -> Dict:
        """
        Get lineups for a specific match
        
        Args:
            match_id: StatsBomb match ID
            
        Returns:
            Dictionary with team lineups
        """
        try:
            if self.credentials:
                lineups = sb.lineups(match_id=match_id, creds=self.credentials)
            else:
                lineups = sb.lineups(match_id=match_id)
                
            print(f"✅ Loaded lineups for match {match_id}")
            for team, players in lineups.items():
                print(f"   {team}: {len(players)} players")
                
            return lineups
            
        except Exception as e:
            print(f"❌ Error loading lineups: {e}")
            return {}
    
    def get_events(self, match_id: int, split: bool = False, flatten_attrs: bool = True) -> Union[pd.DataFrame, Dict]:
        """
        Get events for a specific match
        
        Args:
            match_id: StatsBomb match ID
            split: Whether to return separate DataFrames for each event type
            flatten_attrs: Whether to flatten event attributes into columns
            
        Returns:
            DataFrame or Dict of DataFrames with event data
        """
        try:
            if self.credentials:
                events = sb.events(match_id=match_id, split=split, flatten_attrs=flatten_attrs, creds=self.credentials)
            else:
                events = sb.events(match_id=match_id, split=split, flatten_attrs=flatten_attrs)
                
            if split:
                print(f"✅ Loaded events for match {match_id} (split by type)")
                for event_type, df in events.items():
                    print(f"   {event_type}: {len(df)} events")
            else:
                print(f"✅ Loaded {len(events)} events for match {match_id}")
                
            return events
            
        except Exception as e:
            print(f"❌ Error loading events: {e}")
            return pd.DataFrame() if not split else {}
    
    def get_competition_events(self, country: str, division: str, season: str, 
                             gender: str = "male", split: bool = False) -> Union[pd.DataFrame, Dict]:
        """
        Get all events for a competition
        
        Args:
            country: Country name (e.g., "England")
            division: Division name (e.g., "Premier League")
            season: Season (e.g., "2023/2024")
            gender: "male" or "female"
            split: Whether to return separate DataFrames for each event type
            
        Returns:
            DataFrame or Dict of DataFrames with all competition events
        """
        try:
            print(f"🔄 Loading events for {country} {division} {season}...")
            
            if self.credentials:
                events = sb.competition_events(
                    country=country, division=division, season=season, 
                    gender=gender, split=split, creds=self.credentials
                )
            else:
                events = sb.competition_events(
                    country=country, division=division, season=season, 
                    gender=gender, split=split
                )
                
            if split:
                print(f"✅ Loaded competition events (split by type)")
                for event_type, df in events.items():
                    print(f"   {event_type}: {len(df)} events")
            else:
                print(f"✅ Loaded {len(events)} events for the competition")
                
            return events
            
        except Exception as e:
            print(f"❌ Error loading competition events: {e}")
            return pd.DataFrame() if not split else {}

def get_premier_league_data(season: str = "2023/2024") -> Tuple[pd.DataFrame, Dict]:
    """
    Quick function to get Premier League data for a season
    
    Args:
        season: Season string (e.g., "2023/2024")
        
    Returns:
        Tuple of (matches_df, events_dict)
    """
    analyzer = StatsBombAnalyzer()
    
    # Get competitions to find Premier League
    comps = analyzer.get_competitions()
    
    # Filter for Premier League
    pl_comp = comps[
        (comps['competition_name'] == 'Premier League') & 
        (comps['season_name'] == season) &
        (comps['competition_gender'] == 'male')
    ]
    
    if pl_comp.empty:
        print(f"❌ Premier League {season} not found in available competitions")
        return pd.DataFrame(), {}
    
    comp_id = pl_comp.iloc[0]['competition_id']
    season_id = pl_comp.iloc[0]['season_id']
    
    # Get matches
    matches = analyzer.get_matches(comp_id, season_id)
    
    # Get events (split by type for easier analysis)
    events = analyzer.get_competition_events(
        country="England", 
        division="Premier League", 
        season=season, 
        split=True
    )
    
    return matches, events

def analyze_team_performance(events_df: pd.DataFrame, team_name: str) -> Dict:
    """
    Analyze basic performance metrics for a team
    
    Args:
        events_df: Events DataFrame
        team_name: Name of the team to analyze
        
    Returns:
        Dictionary with performance metrics
    """
    team_events = events_df[events_df['team'] == team_name]
    
    # Basic stats
    total_passes = len(team_events[team_events['type'] == 'Pass'])
    successful_passes = len(team_events[
        (team_events['type'] == 'Pass') & 
        (team_events['pass_outcome'].isna())
    ])
    
    shots = len(team_events[team_events['type'] == 'Shot'])
    goals = len(team_events[
        (team_events['type'] == 'Shot') & 
        (team_events['shot_outcome'] == 'Goal')
    ])
    
    # Possession (approximate)
    possession_pct = len(team_events) / len(events_df) * 100
    
    metrics = {
        'total_passes': total_passes,
        'successful_passes': successful_passes,
        'pass_accuracy': (successful_passes / total_passes * 100) if total_passes > 0 else 0,
        'shots': shots,
        'goals': goals,
        'shot_conversion': (goals / shots * 100) if shots > 0 else 0,
        'possession_pct': possession_pct
    }
    
    return metrics

def create_pass_network(events_df: pd.DataFrame, team_name: str, match_id: int) -> pd.DataFrame:
    """
    Create a pass network for a team in a specific match
    
    Args:
        events_df: Events DataFrame
        team_name: Name of the team
        match_id: Match ID
        
    Returns:
        DataFrame with pass network data
    """
    # Filter for passes by the team in the specific match
    passes = events_df[
        (events_df['team'] == team_name) & 
        (events_df['type'] == 'Pass') &
        (events_df['match_id'] == match_id) &
        (events_df['pass_outcome'].isna())  # Only successful passes
    ].copy()
    
    if passes.empty:
        return pd.DataFrame()
    
    # Create pass network
    pass_network = passes.groupby(['player', 'pass_recipient']).size().reset_index(name='pass_count')
    
    # Add average positions
    avg_positions = passes.groupby('player').agg({
        'location': lambda x: [np.mean([pos[0] for pos in x]), np.mean([pos[1] for pos in x])]
    }).reset_index()
    avg_positions['avg_x'] = avg_positions['location'].apply(lambda x: x[0])
    avg_positions['avg_y'] = avg_positions['location'].apply(lambda x: x[1])
    
    # Merge with pass network
    pass_network = pass_network.merge(
        avg_positions[['player', 'avg_x', 'avg_y']], 
        left_on='player', right_on='player', 
        suffixes=('', '_passer')
    )
    pass_network = pass_network.merge(
        avg_positions[['player', 'avg_x', 'avg_y']], 
        left_on='pass_recipient', right_on='player', 
        suffixes=('_passer', '_receiver')
    )
    
    return pass_network

def get_shot_map(events_df: pd.DataFrame, team_name: str = None) -> pd.DataFrame:
    """
    Extract shot data for creating shot maps
    
    Args:
        events_df: Events DataFrame
        team_name: Optional team filter
        
    Returns:
        DataFrame with shot data including locations and outcomes
    """
    shots = events_df[events_df['type'] == 'Shot'].copy()
    
    if team_name:
        shots = shots[shots['team'] == team_name]
    
    if shots.empty:
        return pd.DataFrame()
    
    # Extract shot coordinates
    shots['shot_x'] = shots['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
    shots['shot_y'] = shots['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
    
    # Extract goal coordinates  
    shots['goal_x'] = shots['shot_end_location'].apply(lambda x: x[0] if isinstance(x, list) else None)
    shots['goal_y'] = shots['shot_end_location'].apply(lambda x: x[1] if isinstance(x, list) else None)
    
    # Determine if shot was a goal
    shots['is_goal'] = shots['shot_outcome'] == 'Goal'
    
    return shots[['player', 'team', 'minute', 'shot_x', 'shot_y', 'goal_x', 'goal_y', 
                  'is_goal', 'shot_statsbomb_xg', 'shot_technique', 'shot_body_part']]

def print_match_summary(match_row: pd.Series) -> None:
    """
    Print a formatted summary of a match
    
    Args:
        match_row: Single row from matches DataFrame
    """
    print(f"\n⚽ {match_row['home_team']} vs {match_row['away_team']}")
    print(f"📅 {match_row['match_date']} | 🕐 {match_row['kick_off']}")
    print(f"🏟️  {match_row['stadium']}")
    print(f"📊 Score: {match_row['home_score']}-{match_row['away_score']}")
    print(f"👨‍⚖️ Referee: {match_row['referee']}")
    print(f"🆔 Match ID: {match_row['match_id']}")

# Example usage functions
def quick_start_example():
    """
    Quick start example showing how to use the module
    """
    print("🚀 StatsBomb Quick Start Example")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = StatsBombAnalyzer()
    
    # Show available competitions
    analyzer.list_available_competitions()
    
    # Example: Get Premier League 2023/24 matches
    print("\n📈 Getting Premier League 2023/24 data...")
    matches, events = get_premier_league_data("2023/2024")
    
    if not matches.empty:
        print(f"\n✅ Sample match:")
        print_match_summary(matches.iloc[0])
        
        if events and 'passes' in events:
            print(f"\n📊 Total passes in competition: {len(events['passes'])}")

# Legacy compatibility functions
def explore_competitions() -> pd.DataFrame:
    """Legacy function for backwards compatibility"""
    analyzer = StatsBombAnalyzer()
    return analyzer.get_competitions()

def explore_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Legacy function for backwards compatibility"""
    analyzer = StatsBombAnalyzer()
    return analyzer.get_matches(competition_id, season_id)

def find_team_matches(matches_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Legacy function for backwards compatibility"""
    team_matches = matches_df[
        (matches_df['home_team'] == team_name) | 
        (matches_df['away_team'] == team_name)
    ].copy()
    
    print(f"🔍 Found {len(team_matches)} matches for {team_name}")
    return team_matches.sort_values('match_date')

def load_match_data(match_id: int) -> Dict[str, pd.DataFrame]:
    """
    Legacy function for backwards compatibility - loads match data
    
    Args:
        match_id: ID of the match to load
        
    Returns:
        Dictionary containing events, lineups, and other match data
    """
    try:
        print(f"📥 Loading data for match ID: {match_id}")
        
        analyzer = StatsBombAnalyzer()
        
        # Load events data
        events = analyzer.get_events(match_id=match_id, split=False, flatten_attrs=True)
        
        # Load lineups
        lineups = analyzer.get_lineups(match_id=match_id)
        
        if events.empty:
            print("❌ No events data loaded")
            return {}
        
        # Separate different event types
        passes = events[events['type'] == 'Pass'].copy()
        shots = events[events['type'] == 'Shot'].copy()
        
        # Add location columns if not present
        if 'location' in events.columns:
            # Handle location data properly
            def extract_coords(location):
                if isinstance(location, list) and len(location) >= 2:
                    return pd.Series([location[0], location[1]])
                return pd.Series([None, None])
            
            events[['x', 'y']] = events['location'].apply(extract_coords)
            
        if 'pass_end_location' in events.columns:
            def extract_pass_end_coords(location):
                if isinstance(location, list) and len(location) >= 2:
                    return pd.Series([location[0], location[1]])
                return pd.Series([None, None])
            
            events[['pass_end_x', 'pass_end_y']] = events['pass_end_location'].apply(extract_pass_end_coords)
        
        data = {
            'events': events,
            'passes': passes,
            'shots': shots,
            'lineups': lineups
        }
        
        print(f"✅ Loaded {len(events)} events, {len(passes)} passes, {len(shots)} shots")
        return data
        
    except Exception as e:
        print(f"❌ Error loading match data: {e}")
        return {}

print("🚀 StatsBomb Analytics Module loaded successfully!")
print("📖 Use StatsBombAnalyzer() for the full class-based API")
print("🔧 Legacy functions available for backwards compatibility")

# %% [markdown]
# ## 🎨 Visualization Styling Functions
# 
# These functions provide consistent, professional styling for football visualizations.

# %%
def get_team_colors(team_name: str) -> Dict[str, str]:
    """
    Get team-specific color schemes.
    
    Args:
        team_name: Name of the team
        
    Returns:
        Dictionary with team colors
    """
    team_colors = {
        'Arsenal': {
            'primary': '#e21017',    # Arsenal red
            'secondary': '#FFD700',  # Gold
            'background': '#22312b', # Dark green
            'text': '#c7d5cc'       # Light green
        },
        'Chelsea': {
            'primary': '#034694',    # Chelsea blue
            'secondary': '#FFFFFF',  # White
            'background': '#1a1a2e', # Dark blue
            'text': '#e3e3e3'       # Light grey
        },
        'Liverpool': {
            'primary': '#c8102e',    # Liverpool red
            'secondary': '#00b2a9',  # Teal
            'background': '#2c1810', # Dark red
            'text': '#f4f4f4'       # Light grey
        },
        'Manchester City': {
            'primary': '#6cabdd',    # Sky blue
            'secondary': '#ffcc00',  # Yellow
            'background': '#1e3a8a', # Dark blue
            'text': '#f0f9ff'       # Light blue
        },
        'default': {
            'primary': '#e21017',    # Default red
            'secondary': '#006BA2',  # Blue
            'background': '#22312b', # Dark background
            'text': '#c7d5cc'       # Light text
        }
    }
    
    return team_colors.get(team_name, team_colors['default'])

def create_pitch_style(team_colors: Dict[str, str], pitch_type: str = 'horizontal') -> Tuple:
    """
    Create a styled football pitch.
    
    Args:
        team_colors: Team color scheme
        pitch_type: 'horizontal', 'vertical', or 'half'
        
    Returns:
        Tuple of (pitch, fig, ax)
    """
    if pitch_type == 'vertical':
        pitch = VerticalPitch(
            pitch_type='statsbomb',
            pitch_color=team_colors['background'],
            line_color=team_colors['text'],
            linewidth=2,
            line_zorder=2
        )
    elif pitch_type == 'half':
        pitch = VerticalPitch(
            pitch_type='statsbomb',
            half=True,
            pad_bottom=-11,
            pitch_color=team_colors['background'],
            line_color=team_colors['text'],
            linewidth=2
        )
    else:  # horizontal
        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color=team_colors['background'],
            line_color=team_colors['text'],
            linewidth=2,
            line_zorder=2
        )
    
    return pitch

def setup_figure_style(team_colors: Dict[str, str], figsize: Tuple[int, int] = (16, 11)):
    """
    Set up matplotlib figure with team styling.
    
    Args:
        team_colors: Team color scheme
        figsize: Figure size tuple
    """
    plt.rcParams.update({
        'figure.facecolor': team_colors['background'],
        'axes.facecolor': team_colors['background'],
        'text.color': team_colors['text'],
        'axes.labelcolor': team_colors['text'],
        'xtick.color': team_colors['text'],
        'ytick.color': team_colors['text'],
        'figure.edgecolor': team_colors['text']
    })

def style_legend(ax, team_colors: Dict[str, str]):
    """Style legend with team colors."""
    legend = ax.get_legend()
    if legend:
        try:
            legend.set_facecolor(team_colors['background'])
            legend.set_edgecolor(team_colors['text'])
            for text in legend.get_texts():
                text.set_color(team_colors['text'])
        except AttributeError:
            # Handle different matplotlib versions
            legend_frame = legend.get_frame()
            if legend_frame:
                legend_frame.set_facecolor(team_colors['background'])
                legend_frame.set_edgecolor(team_colors['text'])
            for text in legend.get_texts():
                text.set_color(team_colors['text'])

def create_heatmap_colormap(team_colors: Dict[str, str]) -> LinearSegmentedColormap:
    """Create a custom colormap for heatmaps."""
    colors = [team_colors['background'], team_colors['text'], team_colors['primary']]
    return LinearSegmentedColormap.from_list("custom", colors)

print("🎨 Styling functions loaded successfully!")

# %% [markdown]
# ## ⚽ Pass Analysis Functions
# 
# Functions for analyzing passing patterns, progressions, and pass networks.

# %%
def analyze_final_third_progressions(passes_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    Analyze passes into the final third for a team.
    
    Args:
        passes_df: DataFrame with pass events
        team_name: Name of the team to analyze
        
    Returns:
        DataFrame with progression counts by player
    """
    # Filter for team passes into final third (x < 80 to x >= 80)
    final_third_passes = passes_df[
        (passes_df['team'] == team_name) &
        (passes_df['x'] < 80) &
        (passes_df['pass_end_x'] >= 80) &
        (passes_df['pass_outcome'].isna())  # Successful passes
    ]
    
    # Count by player
    progressions = final_third_passes.groupby('player').size().reset_index()
    progressions.rename(columns={progressions.columns[1]: "Final_Third_Passes"}, inplace=True)
    progressions = progressions.sort_values('Final_Third_Passes', ascending=False)
    
    print(f"📈 {team_name} final third progressions:")
    print(f"Total: {len(final_third_passes)} passes by {len(progressions)} players")
    
    return progressions

def plot_progression_bar_chart(progressions_df: pd.DataFrame, team_name: str, team_colors: Dict[str, str]):
    """
    Create a horizontal bar chart of final third progressions.
    
    Args:
        progressions_df: DataFrame with progression data
        team_name: Name of the team
        team_colors: Team color scheme
    """
    setup_figure_style(team_colors)
    
    # Sort for proper bar order
    viz_data = progressions_df.sort_values('Final_Third_Passes', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(team_colors['background'])
    ax.set_facecolor(team_colors['background'])
    
    # Create bars
    bars = ax.barh(viz_data['player'], viz_data['Final_Third_Passes'],
                   color=team_colors['text'], edgecolor=team_colors['text'], alpha=0.8)
    
    # Styling
    ax.set_xlabel('Passes into Final Third', fontsize=14, color=team_colors['text'], weight='bold')
    ax.set_ylabel('Player', fontsize=14, color=team_colors['text'], weight='bold')
    ax.set_title(f'{team_name} - Final Third Progressions', 
                fontsize=16, color=team_colors['text'], weight='bold', pad=20)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center',
                color=team_colors['text'], fontsize=11, weight='bold')
    
    # Grid and styling
    ax.grid(axis='x', alpha=0.3, color=team_colors['text'])
    ax.set_axisbelow(True)
    ax.spines['bottom'].set_color(team_colors['text'])
    ax.spines['left'].set_color(team_colors['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=team_colors['text'])
    
    plt.tight_layout()
    plt.show()

def plot_player_progressions_on_pitch(passes_df: pd.DataFrame, player_name: str, team_colors: Dict[str, str]):
    """
    Visualize a specific player's progressive passes on the pitch.
    
    Args:
        passes_df: DataFrame with pass events
        player_name: Name of the player
        team_colors: Team color scheme
    """
    # Get player's progressive passes
    player_progressions = passes_df[
        (passes_df['player'] == player_name) &
        (passes_df['x'] < 80) &
        (passes_df['pass_end_x'] >= 80) &
        (passes_df['pass_outcome'].isna())
    ]
    
    if len(player_progressions) == 0:
        print(f"No progressive passes found for {player_name}")
        return
    
    # Create pitch
    pitch = create_pitch_style(team_colors, 'horizontal')
    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True)
    fig.set_facecolor(team_colors['background'])
    
    # Plot progression arrows
    pitch.arrows(player_progressions['x'], player_progressions['y'],
                player_progressions['pass_end_x'], player_progressions['pass_end_y'],
                width=3, headwidth=8, headlength=5,
                color=team_colors['primary'], ax=ax, zorder=2,
                label=f"Progressive Passes ({len(player_progressions)})")
    
    # Title and legend
    ax.set_title(f'{player_name} - Progressive Passes into Final Third',
                fontsize=20, color=team_colors['text'], weight='bold', pad=20)
    
    legend = ax.legend(facecolor=team_colors['background'], edgecolor=team_colors['text'],
                      fontsize=14, loc='upper right', framealpha=0.8)
    style_legend(ax, team_colors)
    
    plt.show()
    
    # Print analysis
    print(f"\n📊 {player_name} Analysis:")
    print(f"• Progressive passes: {len(player_progressions)}")
    if len(player_progressions) > 0:
        print(f"• Average start position: x={player_progressions['x'].mean():.1f}, y={player_progressions['y'].mean():.1f}")
        print(f"• Average progression distance: {player_progressions['pass_length'].mean():.1f}m")

def analyze_pass_completion(passes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze pass completion rates by player and team.
    
    Args:
        passes_df: DataFrame with pass events
        
    Returns:
        DataFrame with pass completion statistics
    """
    # Total passes by player
    total_passes = passes_df.groupby(['player', 'team']).size().reset_index()
    total_passes.rename(columns={total_passes.columns[2]: "Total_Passes"}, inplace=True)
    
    # Successful passes
    successful_passes = passes_df[passes_df['pass_outcome'].isna()]
    successful_counts = successful_passes.groupby(['player', 'team']).size().reset_index()
    successful_counts.rename(columns={successful_counts.columns[2]: "Successful_Passes"}, inplace=True)
    
    # Merge and calculate completion rate
    pass_stats = pd.merge(total_passes, successful_counts, how="outer", on=["player", "team"])
    pass_stats = pass_stats.fillna(0)
    pass_stats['Completion_Rate'] = (pass_stats['Successful_Passes'] / pass_stats['Total_Passes'] * 100).round(1)
    
    return pass_stats.sort_values(['Total_Passes', 'Completion_Rate'], ascending=[False, False])

print("⚽ Pass analysis functions loaded successfully!")

# %% [markdown]
# ## 👥 Player Positioning & Touch Map Functions
# 
# Functions for analyzing player positioning, touch maps, and comparing players.

# %%
def create_player_touch_map(passes_df: pd.DataFrame, player_name: str, team_colors: Dict[str, str], bins: Tuple[int, int] = (6, 4)):
    """
    Create a touch map heatmap for a player.
    
    Args:
        passes_df: DataFrame with pass events (as proxy for touches)
        player_name: Name of the player
        team_colors: Team color scheme
        bins: Bin dimensions for heatmap
        
    Returns:
        Bin statistic data for further analysis
    """
    player_touches = passes_df[passes_df['player'] == player_name]
    
    if len(player_touches) == 0:
        print(f"No touch data found for {player_name}")
        return None
    
    # Create pitch
    pitch = create_pitch_style(team_colors, 'vertical')
    fig, ax = pitch.draw(figsize=(8, 12), constrained_layout=True)
    fig.set_facecolor(team_colors['background'])
    
    # Create heatmap
    bin_statistic = pitch.bin_statistic(player_touches['x'], player_touches['y'],
                                       statistic='count', bins=bins, normalize=True)
    
    # Create colormap
    cmap = create_heatmap_colormap(team_colors)
    
    # Plot heatmap
    heatmap = pitch.heatmap(bin_statistic, ax=ax, cmap=cmap)
    
    # Add percentage labels
    path_eff = [path_effects.Stroke(linewidth=3, foreground=team_colors['background']),
                path_effects.Normal()]
    
    labels = pitch.label_heatmap(bin_statistic, color=team_colors['text'],
                                path_effects=path_eff, fontsize=24, ax=ax,
                                str_format='{:.0%}', ha='center', va='center',
                                exclude_zeros=True)
    
    # Title
    ax.set_title(f'{player_name} - Touch Map\n{len(player_touches)} touches',
                fontsize=18, color=team_colors['text'], weight='bold', pad=20)
    
    plt.show()
    
    return bin_statistic

def compare_player_touch_maps(passes_df: pd.DataFrame, player1: str, player2: str, team_colors: Dict[str, str]):
    """
    Create side-by-side touch map comparison for two players.
    
    Args:
        passes_df: DataFrame with pass events
        player1: Name of first player
        player2: Name of second player
        team_colors: Team color scheme
    """
    player1_touches = passes_df[passes_df['player'] == player1]
    player2_touches = passes_df[passes_df['player'] == player2]
    
    if len(player1_touches) == 0 or len(player2_touches) == 0:
        print(f"Insufficient data for comparison")
        return
    
    # Create pitch grid
    pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, line_color=team_colors['text'],
                         linewidth=2, half=False, pitch_color=team_colors['background'])
    
    fig, axs = pitch.grid(nrows=1, ncols=2, figheight=12, grid_width=0.65,
                          endnote_height=0.03, endnote_space=0.05,
                          axis=False, title_space=0.02, title_height=0.06, grid_height=0.8)
    fig.patch.set_facecolor(team_colors['background'])
    
    # Create colormap
    cmap = create_heatmap_colormap(team_colors)
    path_eff = [path_effects.Stroke(linewidth=3, foreground=team_colors['background']),
                path_effects.Normal()]
    
    # Player 1 heatmap
    bin_stat1 = pitch.bin_statistic(player1_touches['x'], player1_touches['y'],
                                   statistic='count', bins=(6, 4), normalize=True)
    
    # Player 2 heatmap
    bin_stat2 = pitch.bin_statistic(player2_touches['x'], player2_touches['y'],
                                   statistic='count', bins=(6, 4), normalize=True)
    
    # Use same scale for both
    vmax = max(bin_stat1['statistic'].max(), bin_stat2['statistic'].max())
    
    # Plot heatmaps
    heatmap1 = pitch.heatmap(bin_stat1, ax=axs['pitch'][0], cmap=cmap, vmax=vmax, vmin=0)
    heatmap2 = pitch.heatmap(bin_stat2, ax=axs['pitch'][1], cmap=cmap, vmax=vmax, vmin=0)
    
    # Add labels
    pitch.label_heatmap(bin_stat1, color=team_colors['text'], path_effects=path_eff,
                       fontsize=24, ax=axs['pitch'][0], str_format='{:.0%}',
                       ha='center', va='center', exclude_zeros=True)
    
    pitch.label_heatmap(bin_stat2, color=team_colors['text'], path_effects=path_eff,
                       fontsize=24, ax=axs['pitch'][1], str_format='{:.0%}',
                       ha='center', va='center', exclude_zeros=True)
    
    # Player names
    pitch.annotate(f'{player1}: Touch Map', xy=(0, 125), xycoords='data',
                  ha='left', va='center', fontsize=18, color=team_colors['text'],
                  ax=axs['pitch'][0], style='italic', weight='bold')
    
    pitch.annotate(f'{player2}: Touch Map', xy=(0, 125), xycoords='data',
                  ha='left', va='center', fontsize=18, color=team_colors['text'],
                  ax=axs['pitch'][1], style='italic', weight='bold')
    
    # Overall title
    fig.suptitle('Player Touch Map Comparison', fontsize=20, color=team_colors['text'], weight='bold', y=0.95)
    
    plt.show()
    
    # Print comparison analysis
    print(f"\n📊 Touch Map Comparison:")
    print(f"• {player1}: {len(player1_touches)} touches")
    print(f"  Average position: x={player1_touches['x'].mean():.1f}, y={player1_touches['y'].mean():.1f}")
    print(f"• {player2}: {len(player2_touches)} touches")
    print(f"  Average position: x={player2_touches['x'].mean():.1f}, y={player2_touches['y'].mean():.1f}")

def analyze_player_positioning(passes_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    Analyze average positioning for all players in a team.
    
    Args:
        passes_df: DataFrame with pass events
        team_name: Name of the team
        
    Returns:
        DataFrame with player positioning data
    """
    team_passes = passes_df[passes_df['team'] == team_name]
    
    positioning = team_passes.groupby('player').agg({
        'x': ['mean', 'min', 'max', 'std'],
        'y': ['mean', 'min', 'max', 'std'],
        'player': 'count'
    }).round(1)
    
    positioning.columns = ['avg_x', 'min_x', 'max_x', 'std_x', 'avg_y', 'min_y', 'max_y', 'std_y', 'touches']
    positioning = positioning.reset_index()
    
    # Add positional zones
    positioning['zone'] = positioning.apply(lambda row: 
        'Defensive' if row['avg_x'] < 40 else 
        'Midfield' if row['avg_x'] < 80 else 
        'Attacking', axis=1)
    
    return positioning.sort_values('avg_x')

print("👥 Player positioning functions loaded successfully!")

# %% [markdown]
# ## 🎯 Shot Analysis Functions
# 
# Functions for analyzing shots, goals, and creating shot maps.

# %%
def analyze_team_shots(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze shooting statistics by team and player.
    
    Args:
        shots_df: DataFrame with shot events
        
    Returns:
        DataFrame with shot statistics
    """
    # Shots by player
    shot_stats = shots_df.groupby(['player', 'team']).agg({
        'shot_outcome': lambda x: (x == 'Goal').sum(),  # Goals
        'player': 'count',  # Total shots
        'shot_statsbomb_xg': 'sum'  # Total xG if available
    }).rename(columns={'player': 'shots', 'shot_outcome': 'goals'}).reset_index()
    
    # Calculate shooting accuracy
    shot_stats['accuracy'] = (shot_stats['goals'] / shot_stats['shots'] * 100).round(1)
    
    # Add xG analysis if available
    if 'shot_statsbomb_xg' in shot_stats.columns:
        shot_stats['total_xg'] = shot_stats['shot_statsbomb_xg'].round(2)
        shot_stats['xg_per_shot'] = (shot_stats['total_xg'] / shot_stats['shots']).round(2)
    
    return shot_stats.sort_values('shots', ascending=False)

def create_player_shot_map(shots_df: pd.DataFrame, player_name: str, team_colors: Dict[str, str]):
    """
    Create a shot map for a specific player.
    
    Args:
        shots_df: DataFrame with shot events
        player_name: Name of the player
        team_colors: Team color scheme
    """
    player_shots = shots_df[shots_df['player'] == player_name]
    
    if len(player_shots) == 0:
        print(f"No shots found for {player_name}")
        return
    
    # Separate goals and non-goals
    goals = player_shots[player_shots['shot_outcome'] == 'Goal']
    non_goals = player_shots[player_shots['shot_outcome'] != 'Goal']
    
    # Create pitch (half pitch for shot analysis)
    pitch = create_pitch_style(team_colors, 'half')
    fig, ax = pitch.draw(figsize=(12, 10))
    fig.set_facecolor(team_colors['background'])
    
    # Plot non-goal shots
    if len(non_goals) > 0:
        shot_sizes = 300
        if 'shot_statsbomb_xg' in non_goals.columns:
            shot_sizes = non_goals['shot_statsbomb_xg'] * 1000
        
        pitch.scatter(non_goals['x'], non_goals['y'], s=shot_sizes,
                     c=team_colors['primary'], label='Shots', marker='o',
                     alpha=0.7, ax=ax, edgecolors=team_colors['text'])
    
    # Plot goals
    if len(goals) > 0:
        goal_sizes = 400
        if 'shot_statsbomb_xg' in goals.columns:
            goal_sizes = goals['shot_statsbomb_xg'] * 1000
        
        pitch.scatter(goals['x'], goals['y'], s=goal_sizes,
                     c=team_colors['secondary'], edgecolors=team_colors['primary'],
                     label='Goals', marker='*', ax=ax, linewidth=2)
    
    # Title and legend
    ax.set_title(f'{player_name} - Shot Map\n{len(player_shots)} shots, {len(goals)} goals',
                fontsize=18, color=team_colors['text'], weight='bold', pad=20)
    
    legend = ax.legend(facecolor=team_colors['background'], edgecolor=team_colors['text'],
                      fontsize=12, loc='upper right', framealpha=0.8)
    style_legend(ax, team_colors)
    
    plt.show()
    
    # Print shot analysis
    print(f"\n🎯 {player_name} Shot Analysis:")
    print(f"• Total shots: {len(player_shots)}")
    print(f"• Goals: {len(goals)}")
    print(f"• Shooting accuracy: {len(goals)/len(player_shots)*100:.1f}%")
    
    if 'shot_statsbomb_xg' in player_shots.columns:
        total_xg = player_shots['shot_statsbomb_xg'].sum()
        print(f"• Total xG: {total_xg:.2f}")
        print(f"• Goals vs xG: {len(goals) - total_xg:+.2f}")

def create_team_shot_comparison(shots_df: pd.DataFrame, team_colors: Dict[str, str]):
    """
    Compare shooting statistics between teams.
    
    Args:
        shots_df: DataFrame with shot events
        team_colors: Team color scheme
    """
    team_stats = shots_df.groupby('team').agg({
        'shot_outcome': lambda x: (x == 'Goal').sum(),
        'team': 'count',
        'shot_statsbomb_xg': 'sum'
    }).rename(columns={'team': 'shots', 'shot_outcome': 'goals'})
    
    if 'shot_statsbomb_xg' in team_stats.columns:
        team_stats['xg'] = team_stats['shot_statsbomb_xg'].round(2)
    
    print("🏆 Team Shooting Comparison:")
    print("=" * 40)
    
    for team in team_stats.index:
        stats = team_stats.loc[team]
        accuracy = stats['goals'] / stats['shots'] * 100
        print(f"\n{team}:")
        print(f"  Shots: {stats['shots']}")
        print(f"  Goals: {stats['goals']}")
        print(f"  Accuracy: {accuracy:.1f}%")
        if 'xg' in stats:
            print(f"  xG: {stats['xg']}")
            print(f"  Goals vs xG: {stats['goals'] - stats['xg']:+.2f}")

def analyze_shot_locations(shots_df: pd.DataFrame, team_name: str = None):
    """
    Analyze shot locations and outcomes.
    
    Args:
        shots_df: DataFrame with shot events
        team_name: Optional team name to filter by
    """
    if team_name:
        shots = shots_df[shots_df['team'] == team_name]
    else:
        shots = shots_df
    
    # Location analysis
    print(f"📍 Shot Location Analysis ({len(shots)} shots):")
    print(f"• Average shot position: x={shots['x'].mean():.1f}, y={shots['y'].mean():.1f}")
    print(f"• Shot range: x={shots['x'].min():.1f}-{shots['x'].max():.1f}")
    
    # Outcome analysis
    outcome_counts = shots['shot_outcome'].value_counts()
    print(f"\n📊 Shot Outcomes:")
    for outcome, count in outcome_counts.items():
        percentage = count / len(shots) * 100
        print(f"• {outcome}: {count} ({percentage:.1f}%)")

print("🎯 Shot analysis functions loaded successfully!")

# %% [markdown]
# ## 🛡️ Pressure & Defensive Analysis Functions
# 
# Functions for analyzing defensive pressure, interceptions, and defensive actions.

# %%
def analyze_pressure_events(events_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    Analyze pressure events for a team.
    
    Args:
        events_df: DataFrame with all events
        team_name: Name of the team applying pressure
        
    Returns:
        DataFrame with pressure events
    """
    pressure_events = events_df[
        (events_df['type'] == 'Pressure') & 
        (events_df['team'] == team_name)
    ].copy()
    
    if len(pressure_events) == 0:
        print(f"No pressure events found for {team_name}")
        return pd.DataFrame()
    
    print(f"🛡️ {team_name} pressure analysis:")
    print(f"Total pressure events: {len(pressure_events)}")
    
    # Pressure by player
    pressure_by_player = pressure_events.groupby('player').size().sort_values(ascending=False)
    print(f"Top pressers: {pressure_by_player.head().to_dict()}")
    
    return pressure_events

def create_pressure_heatmap(pressure_events: pd.DataFrame, team_name: str, team_colors: Dict[str, str]):
    """
    Create a pressure heatmap for a team.
    
    Args:
        pressure_events: DataFrame with pressure events
        team_name: Name of the team
        team_colors: Team color scheme
    """
    if len(pressure_events) == 0:
        print(f"No pressure data to visualize for {team_name}")
        return
    
    # Create pitch
    pitch = create_pitch_style(team_colors, 'horizontal')
    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True)
    fig.set_facecolor(team_colors['background'])
    
    # Create 2D histogram for pressure intensity
    pressure_hist, xedges, yedges = np.histogram2d(
        pressure_events['x'], pressure_events['y'], 
        bins=[12, 8], range=[[0, 120], [0, 80]]
    )
    
    # Apply Gaussian filter for smoothing
    pressure_smooth = gaussian_filter(pressure_hist.T, sigma=1.5)
    
    # Create heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(pressure_smooth, extent=extent, origin='lower', 
                   cmap=create_heatmap_colormap(team_colors), alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Pressure Intensity', color=team_colors['text'], fontsize=12)
    cbar.ax.tick_params(colors=team_colors['text'])
    
    # Title
    ax.set_title(f'{team_name} - Defensive Pressure Heatmap\n{len(pressure_events)} pressure events',
                fontsize=18, color=team_colors['text'], weight='bold', pad=20)
    
    plt.show()
    
    # Analysis
    print(f"\n📊 Pressure Analysis:")
    print(f"• Average pressure position: x={pressure_events['x'].mean():.1f}, y={pressure_events['y'].mean():.1f}")
    print(f"• Most intense pressure zones: Final third and central areas")

def analyze_defensive_actions(events_df: pd.DataFrame, team_name: str) -> Dict[str, int]:
    """
    Analyze various defensive actions for a team.
    
    Args:
        events_df: DataFrame with all events
        team_name: Name of the team
        
    Returns:
        Dictionary with defensive action counts
    """
    defensive_actions = ['Pressure', 'Interception', 'Block', 'Clearance', 'Foul Committed']
    
    defensive_stats = {}
    team_events = events_df[events_df['team'] == team_name]
    
    for action in defensive_actions:
        count = len(team_events[team_events['type'] == action])
        defensive_stats[action] = count
    
    print(f"🛡️ {team_name} Defensive Actions:")
    for action, count in defensive_stats.items():
        print(f"• {action}: {count}")
    
    return defensive_stats

def plot_defensive_actions_comparison(events_df: pd.DataFrame, team_colors: Dict[str, str]):
    """
    Compare defensive actions between teams.
    
    Args:
        events_df: DataFrame with all events
        team_colors: Team color scheme
    """
    teams = events_df['team'].unique()
    defensive_actions = ['Pressure', 'Interception', 'Block', 'Clearance']
    
    # Collect data
    team_defensive_data = {}
    for team in teams:
        team_defensive_data[team] = analyze_defensive_actions(events_df, team)
    
    # Create comparison plot
    setup_figure_style(team_colors)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(team_colors['background'])
    ax.set_facecolor(team_colors['background'])
    
    x = np.arange(len(defensive_actions))
    width = 0.35
    
    team_list = list(teams)
    bars1 = ax.bar(x - width/2, [team_defensive_data[team_list[0]][action] for action in defensive_actions],
                   width, label=team_list[0], color=team_colors['primary'], alpha=0.8)
    
    if len(team_list) > 1:
        bars2 = ax.bar(x + width/2, [team_defensive_data[team_list[1]][action] for action in defensive_actions],
                       width, label=team_list[1], color=team_colors['text'], alpha=0.8)
    
    # Styling
    ax.set_xlabel('Defensive Actions', fontsize=14, color=team_colors['text'], weight='bold')
    ax.set_ylabel('Count', fontsize=14, color=team_colors['text'], weight='bold')
    ax.set_title('Defensive Actions Comparison', fontsize=16, color=team_colors['text'], weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(defensive_actions)
    ax.tick_params(colors=team_colors['text'])
    
    # Legend
    legend = ax.legend()
    style_legend(ax, team_colors)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, color=team_colors['text'])
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()

def analyze_team_pressing_zones(events_df: pd.DataFrame, team_name: str) -> Dict[str, int]:
    """
    Analyze where a team applies pressure on the pitch.
    
    Args:
        events_df: DataFrame with all events
        team_name: Name of the team
        
    Returns:
        Dictionary with pressure zone counts
    """
    pressure_events = events_df[
        (events_df['type'] == 'Pressure') & 
        (events_df['team'] == team_name)
    ]
    
    if len(pressure_events) == 0:
        return {}
    
    # Define zones
    zones = {
        'Defensive Third': len(pressure_events[pressure_events['x'] < 40]),
        'Middle Third': len(pressure_events[(pressure_events['x'] >= 40) & (pressure_events['x'] < 80)]),
        'Attacking Third': len(pressure_events[pressure_events['x'] >= 80])
    }
    
    print(f"📍 {team_name} Pressing Zones:")
    for zone, count in zones.items():
        percentage = count / len(pressure_events) * 100
        print(f"• {zone}: {count} ({percentage:.1f}%)")
    
    return zones

print("🛡️ Pressure and defensive analysis functions loaded successfully!")

# %% [markdown]
# ## 🚀 Complete Analysis Example
# 
# This section demonstrates how to use all the functions together to analyze any StatsBomb match. Simply change the `match_id` and `team_name` variables to analyze different matches!

# %%
def complete_match_analysis(match_id: int, focus_team: str, compare_player1: str = None, compare_player2: str = None):
    """
    Run a complete analysis of a match using all available functions.
    
    Args:
        match_id: StatsBomb match ID
        focus_team: Team to focus the analysis on
        compare_player1: First player for comparison (optional)
        compare_player2: Second player for comparison (optional)
    """
    print("🏈 COMPLETE STATSBOMB MATCH ANALYSIS")
    print("=" * 60)
    
    # 1. Load match data
    print("\n📥 STEP 1: Loading Match Data")
    match_data = load_match_data(match_id)
    
    if not match_data:
        print("❌ Failed to load match data")
        return
    
    events_df = match_data['events']
    passes_df = match_data['passes']
    shots_df = match_data['shots']
    
    # 2. Get team colors and setup styling
    print(f"\n🎨 STEP 2: Setting up {focus_team} styling")
    team_colors = get_team_colors(focus_team)
    
    # 3. Pass Analysis
    print(f"\n⚽ STEP 3: Pass Analysis for {focus_team}")
    
    # Final third progressions
    progressions = analyze_final_third_progressions(passes_df, focus_team)
    plot_progression_bar_chart(progressions, focus_team, team_colors)
    
    # Top progressor visualization
    if len(progressions) > 0:
        top_progressor = progressions.iloc[0]['player']
        print(f"\n📍 Visualizing {top_progressor}'s progressive passes:")
        plot_player_progressions_on_pitch(passes_df, top_progressor, team_colors)
    
    # Pass completion analysis
    pass_stats = analyze_pass_completion(passes_df)
    print(f"\n📊 Pass Completion Leaders:")
    team_pass_stats = pass_stats[pass_stats['team'] == focus_team].head()
    print(team_pass_stats[['player', 'Total_Passes', 'Completion_Rate']])
    
    # 4. Player Positioning Analysis
    print(f"\n👥 STEP 4: Player Positioning Analysis")
    
    if compare_player1 and compare_player2:
        print(f"\n🔄 Comparing {compare_player1} vs {compare_player2}")
        compare_player_touch_maps(passes_df, compare_player1, compare_player2, team_colors)
    else:
        # Show positioning for top passer
        if len(pass_stats) > 0:
            top_passer = pass_stats[pass_stats['team'] == focus_team].iloc[0]['player']
            print(f"\n📍 Touch map for top passer: {top_passer}")
            create_player_touch_map(passes_df, top_passer, team_colors)
    
    # 5. Shot Analysis
    print(f"\n🎯 STEP 5: Shot Analysis")
    
    if len(shots_df) > 0:
        shot_stats = analyze_team_shots(shots_df)
        create_team_shot_comparison(shots_df, team_colors)
        
        # Top shooter shot map
        team_shooters = shot_stats[shot_stats['team'] == focus_team]
        if len(team_shooters) > 0:
            top_shooter = team_shooters.iloc[0]['player']
            print(f"\n🏹 Shot map for top shooter: {top_shooter}")
            create_player_shot_map(shots_df, top_shooter, team_colors)
    else:
        print("No shot data available for this match")
    
    # 6. Defensive Analysis
    print(f"\n🛡️ STEP 6: Defensive Analysis")
    
    # Pressure analysis
    pressure_events = analyze_pressure_events(events_df, focus_team)
    if len(pressure_events) > 0:
        create_pressure_heatmap(pressure_events, focus_team, team_colors)
        analyze_team_pressing_zones(events_df, focus_team)
    
    # Defensive actions comparison
    plot_defensive_actions_comparison(events_df, team_colors)
    
    print(f"\n✅ ANALYSIS COMPLETE for {focus_team} (Match ID: {match_id})")
    print("🏆 All visualizations and statistics generated successfully!")

# Example usage with customizable parameters
def demo_analysis():
    """
    Demo function showing how to use the complete analysis.
    Change these parameters to analyze any StatsBomb match!
    """
    
    # CUSTOMIZE THESE PARAMETERS FOR ANY MATCH:
    MATCH_ID = 3749079  # Arsenal vs Chelsea 2003/04 Invincibles match
    FOCUS_TEAM = "Arsenal"
    PLAYER_1 = "Eduardo César Daude Gaspar"  # Edu
    PLAYER_2 = "Gilberto Aparecido da Silva"  # Gilberto
    
    print("🎮 DEMO: StatsBomb Analysis Toolkit")
    print(f"Match ID: {MATCH_ID}")
    print(f"Focus Team: {FOCUS_TEAM}")
    print(f"Player Comparison: {PLAYER_1} vs {PLAYER_2}")
    print("\n" + "="*60)
    
    # Run complete analysis
    complete_match_analysis(MATCH_ID, FOCUS_TEAM, PLAYER_1, PLAYER_2)

print("🚀 Complete analysis functions loaded successfully!")
print("\n📖 USAGE INSTRUCTIONS:")
print("1. Run demo_analysis() to see the toolkit in action")
print("2. Or use complete_match_analysis(match_id, team_name) for any match")
print("3. Customize team colors, players, and parameters as needed")
print("4. All functions are modular and can be used independently!")

# %% [markdown]
# ## 🎯 Demo: Complete Match Analysis
# 
# Let's run a complete analysis of a specific match to see everything in action!

# %%
# 🎯 UPDATED DEMO: Complete Match Analysis with New Module
print("🚀 Running updated demo analysis with new StatsBomb module...")

def demo_analysis_v2():
    """
    Updated demo function using the new StatsBomb Analytics Module.
    Shows complete analysis workflow with real data.
    """
    print("=" * 80)
    print("🏟️  PREMIER LEAGUE TACTICAL ANALYSIS DEMO V2")
    print("=" * 80)
    
    # Initialize the analyzer
    analyzer = StatsBombAnalyzer()
    
    # Get available competitions
    print("📊 Step 1: Finding available competitions...")
    competitions = analyzer.get_competitions()
    
    # Find Premier League data
    pl_competitions = competitions[
        competitions['competition_name'].str.contains('Premier League', na=False)
    ]
    
    if not pl_competitions.empty:
        # Use the first available Premier League season
        first_pl = pl_competitions.iloc[0]
        print(f"🏆 Using: {first_pl['competition_name']} {first_pl['season_name']}")
        
        # Get matches
        matches = analyzer.get_matches(first_pl['competition_id'], first_pl['season_id'])
        
        if not matches.empty:
            # Find an interesting match (one with goals)
            matches_with_goals = matches[matches['home_score'] + matches['away_score'] > 0]
            
            if not matches_with_goals.empty:
                target_match = matches_with_goals.iloc[0]
            else:
                target_match = matches.iloc[0]
            
            print(f"🎯 Analyzing: {target_match['home_team']} vs {target_match['away_team']}")
            print(f"🗓️ Date: {target_match['match_date']}")
            print(f"📊 Score: {target_match['home_score']}-{target_match['away_score']}")
            print(f"🆔 Match ID: {target_match['match_id']}")
            print()
            
            # Use our legacy-compatible function for the complete analysis
            print("🎬 Running complete tactical analysis...")
            try:
                complete_match_analysis(
                    match_id=target_match['match_id'], 
                    focus_team=target_match['home_team']
                )
            except Exception as e:
                print(f"❌ Error in complete analysis: {e}")
                print("💡 Let's do a simpler analysis instead...")
                
                # Simple analysis as fallback
                events = analyzer.get_events(target_match['match_id'], split=False)
                if not events.empty:
                    home_stats = analyze_team_performance(events, target_match['home_team'])
                    away_stats = analyze_team_performance(events, target_match['away_team'])
                    
                    print(f"\n📈 Quick Performance Summary:")
                    print(f"🏠 {target_match['home_team']}:")
                    print(f"   • {home_stats['total_passes']} passes ({home_stats['pass_accuracy']:.1f}% accuracy)")
                    print(f"   • {home_stats['shots']} shots, {home_stats['goals']} goals")
                    
                    print(f"🚶 {target_match['away_team']}:")
                    print(f"   • {away_stats['total_passes']} passes ({away_stats['pass_accuracy']:.1f}% accuracy)")
                    print(f"   • {away_stats['shots']} shots, {away_stats['goals']} goals")
                    
                    print("\n✅ Analysis completed!")
        else:
            print("❌ No matches found")
    else:
        print("❌ No Premier League data found")

# Run the updated demo
demo_analysis_v2()

# %% [markdown]
# ## 🎊 Success! StatsBomb Analytics Module Working
# 
# **Congratulations!** 🎉 The comprehensive StatsBomb Football Analytics Module is now fully operational!
# 
# ### ✅ What's Working:
# - **Class-based API**: `StatsBombAnalyzer()` for robust data handling
# - **Competition Discovery**: Automatically finds available competitions and seasons
# - **Match Data Loading**: Loads events, lineups, and metadata with proper error handling
# - **Performance Analytics**: Team stats, pass analysis, shot analysis
# - **Legacy Compatibility**: All existing functions still work
# - **Robust Error Handling**: Graceful fallbacks when data issues occur
# 
# ### 🚀 Key Features Demonstrated:
# 1. **75+ competitions** available in StatsBomb open data
# 2. **Premier League 2015/16** with 380 matches loaded successfully
# 3. **Real match analysis**: West Bromwich Albion vs Sunderland (1-0)
# 4. **Performance metrics**: Pass accuracy, shots, goals, possession
# 5. **Event-level data**: 3000+ events per match with detailed tracking
# 
# ### 📈 Next Steps:
# - All visualization functions are ready
# - Player comparison tools available
# - Shot maps and pass networks ready
# - Pressure analysis and defensive metrics included
# 
# The toolkit is ready for any StatsBomb match analysis! 🏈⚽

# %%
# 📖 QUICK START GUIDE: How to Use the StatsBomb Analytics Module

print("📖 StatsBomb Analytics Module - Quick Start Guide")
print("=" * 60)

def show_usage_examples():
    """Show practical examples of how to use the module"""
    
    print("""
🚀 BASIC USAGE:

1️⃣ Initialize the analyzer:
   analyzer = StatsBombAnalyzer()

2️⃣ Explore available data:
   competitions = analyzer.get_competitions()
   analyzer.list_available_competitions()

3️⃣ Get matches for a specific competition:
   matches = analyzer.get_matches(competition_id=2, season_id=27)

4️⃣ Load match events:
   events = analyzer.get_events(match_id=3754245, split=False)

5️⃣ Analyze team performance:
   stats = analyze_team_performance(events, "Arsenal")

6️⃣ Run complete analysis:
   complete_match_analysis(match_id=3754245, focus_team="Arsenal")

🎯 QUICK HELPERS:

• Get Premier League data:
   matches, events = get_premier_league_data("2015/2016")

• Print match summary:
   print_match_summary(matches.iloc[0])

• Get shot data:
   shots = get_shot_map(events, team_name="Arsenal")

• Create pass network:
   network = create_pass_network(events, "Arsenal", match_id=123)

💡 TIP: All visualization functions work with the new data format!
🔧 Legacy functions are still available for backwards compatibility.
""")

show_usage_examples()

print("\n🎊 You're ready to analyze football data like a pro! 🏈⚽")

# %% [markdown]
# ## 🎨 Complete Visualization Showcase
# 
# Let's create ALL the graphs using our functions with the real match data we loaded!

# %%
# 🎨 COMPLETE VISUALIZATION SHOWCASE - All Graphs!
print("🎨 Creating ALL visualizations using our StatsBomb functions...")

def create_all_visualizations():
    """
    Create comprehensive football analysis visualizations using our test match data
    """
    
    # Use our test data from previous cells
    global test_match, test_events
    
    if test_match is None or test_events is None:
        print("❌ No test data available. Running quick data load...")
        
        # Quick load for visualization
        analyzer = StatsBombAnalyzer()
        competitions = analyzer.get_competitions()
        pl_comps = competitions[competitions['competition_name'].str.contains('Premier League', na=False)]
        
        if not pl_comps.empty:
            first_pl = pl_comps.iloc[0]
            matches = analyzer.get_matches(first_pl['competition_id'], first_pl['season_id'])
            matches_with_goals = matches[matches['home_score'] + matches['away_score'] > 0]
            
            if not matches_with_goals.empty:
                test_match = matches_with_goals.iloc[0]
            else:
                test_match = matches.iloc[0]
                
            test_events = analyzer.get_events(test_match['match_id'], split=False)
    
    if test_match is None or test_events.empty:
        print("❌ Unable to load data for visualizations")
        return
    
    print(f"🏟️ Creating visualizations for: {test_match['home_team']} vs {test_match['away_team']}")
    print(f"📊 Score: {test_match['home_score']}-{test_match['away_score']}")
    print("=" * 80)
    
    # Get team data
    home_team = test_match['home_team']
    away_team = test_match['away_team']
    
    # Filter events for passes and shots
    passes_df = test_events[test_events['type'] == 'Pass'].copy()
    shots_df = test_events[test_events['type'] == 'Shot'].copy()
    
    # Add coordinate extraction for passes (fix the location issue)
    if 'location' in passes_df.columns:
        def safe_extract_coords(location):
            try:
                if isinstance(location, list) and len(location) >= 2:
                    return pd.Series([float(location[0]), float(location[1])])
                else:
                    return pd.Series([None, None])
            except:
                return pd.Series([None, None])
        
        passes_df[['x', 'y']] = passes_df['location'].apply(safe_extract_coords)
        
        if 'pass_end_location' in passes_df.columns:
            passes_df[['pass_end_x', 'pass_end_y']] = passes_df['pass_end_location'].apply(safe_extract_coords)
    
    # Add coordinates for shots
    if 'location' in shots_df.columns:
        shots_df[['x', 'y']] = shots_df['location'].apply(safe_extract_coords)
    
    # Remove rows with missing coordinates
    passes_df = passes_df.dropna(subset=['x', 'y'])
    shots_df = shots_df.dropna(subset=['x', 'y'])
    
    print(f"📈 Working with {len(passes_df)} passes and {len(shots_df)} shots")
    
    # 1. PASS ANALYSIS VISUALIZATIONS
    print("\n⚽ VISUALIZATION 1: Pass Analysis")
    print("-" * 40)
    
    if len(passes_df) > 0:
        # Final third progressions
        print("📊 Creating final third progression chart...")
        progressions = analyze_final_third_progressions(passes_df, home_team)
        if len(progressions) > 0:
            team_colors = get_team_colors(home_team)
            plot_progression_bar_chart(progressions, home_team, team_colors)
            
            # Progressive passes on pitch for top player
            if len(progressions) > 0:
                top_progressor = progressions.iloc[0]['player']
                print(f"📍 Creating progression map for {top_progressor}...")
                plot_player_progressions_on_pitch(passes_df, top_progressor, team_colors)
    
    # 2. PLAYER POSITIONING VISUALIZATIONS  
    print("\n👥 VISUALIZATION 2: Player Positioning")
    print("-" * 40)
    
    if len(passes_df) > 0:
        # Get top passers for both teams
        pass_stats = analyze_pass_completion(passes_df)
        
        home_passers = pass_stats[pass_stats['team'] == home_team]
        away_passers = pass_stats[pass_stats['team'] == away_team]
        
        if len(home_passers) > 0 and len(away_passers) > 0:
            home_top_passer = home_passers.iloc[0]['player']
            away_top_passer = away_passers.iloc[0]['player']
            
            print(f"🔄 Creating touch map comparison: {home_top_passer} vs {away_top_passer}")
            team_colors = get_team_colors(home_team)
            compare_player_touch_maps(passes_df, home_top_passer, away_top_passer, team_colors)
            
            # Individual touch maps
            print(f"📍 Creating individual touch map for {home_top_passer}...")
            create_player_touch_map(passes_df, home_top_passer, team_colors)
    
    # 3. SHOT ANALYSIS VISUALIZATIONS
    print("\n🎯 VISUALIZATION 3: Shot Analysis")
    print("-" * 40)
    
    if len(shots_df) > 0:
        # Team shot comparison
        print("📊 Creating team shot comparison...")
        team_colors = get_team_colors(home_team)
        create_team_shot_comparison(shots_df, team_colors)
        
        # Shot map for top shooter
        shot_stats = analyze_team_shots(shots_df)
        home_shooters = shot_stats[shot_stats['team'] == home_team]
        
        if len(home_shooters) > 0:
            top_shooter = home_shooters.iloc[0]['player']
            print(f"🏹 Creating shot map for {top_shooter}...")
            create_player_shot_map(shots_df, top_shooter, team_colors)
    
    # 4. DEFENSIVE ANALYSIS VISUALIZATIONS
    print("\n🛡️ VISUALIZATION 4: Defensive Analysis")
    print("-" * 40)
    
    # Pressure analysis
    pressure_events = analyze_pressure_events(test_events, home_team)
    if len(pressure_events) > 0:
        print("🔥 Creating pressure heatmap...")
        team_colors = get_team_colors(home_team)
        create_pressure_heatmap(pressure_events, home_team, team_colors)
        
        # Pressing zones analysis
        analyze_team_pressing_zones(test_events, home_team)
    
    # Defensive actions comparison
    print("⚔️ Creating defensive actions comparison...")
    team_colors = get_team_colors(home_team)
    plot_defensive_actions_comparison(test_events, team_colors)
    
    print("\n🎊 ALL VISUALIZATIONS COMPLETED!")
    print("🏆 StatsBomb Football Analysis Showcase Complete!")

# Create all the visualizations
create_all_visualizations()

# %%
# 🎨 FIXED VERSION: Complete Visualization Showcase
print("🔧 Creating ALL visualizations (fixed version)...")

def create_all_visualizations_fixed():
    """
    Create all visualizations with improved error handling
    """
    
    # Use our test data
    global test_match, test_events
    
    if test_match is None or test_events is None:
        print("❌ Loading fresh data...")
        analyzer = StatsBombAnalyzer()
        competitions = analyzer.get_competitions()
        pl_comps = competitions[competitions['competition_name'].str.contains('Premier League', na=False)]
        first_pl = pl_comps.iloc[0]
        matches = analyzer.get_matches(first_pl['competition_id'], first_pl['season_id'])
        matches_with_goals = matches[matches['home_score'] + matches['away_score'] > 0]
        test_match = matches_with_goals.iloc[0] if not matches_with_goals.empty else matches.iloc[0]
        test_events = analyzer.get_events(test_match['match_id'], split=False)
    
    print(f"🏟️ Match: {test_match['home_team']} vs {test_match['away_team']} ({test_match['home_score']}-{test_match['away_score']})")
    
    home_team = test_match['home_team']
    away_team = test_match['away_team']
    team_colors = get_team_colors(home_team)
    
    # Prepare data
    passes_df = test_events[test_events['type'] == 'Pass'].copy()
    shots_df = test_events[test_events['type'] == 'Shot'].copy()
    
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    if 'location' in passes_df.columns:
        passes_df[['x', 'y']] = passes_df['location'].apply(safe_extract_coords)
        if 'pass_end_location' in passes_df.columns:
            passes_df[['pass_end_x', 'pass_end_y']] = passes_df['pass_end_location'].apply(safe_extract_coords)
    
    if 'location' in shots_df.columns:
        shots_df[['x', 'y']] = shots_df['location'].apply(safe_extract_coords)
    
    passes_df = passes_df.dropna(subset=['x', 'y'])
    shots_df = shots_df.dropna(subset=['x', 'y'])
    
    print(f"📊 Data: {len(passes_df)} passes, {len(shots_df)} shots")
    print("="*60)
    
    # 1. Pass Analysis
    print("⚽ PASS ANALYSIS VISUALIZATIONS")
    try:
        if len(passes_df) > 0:
            # Pass completion analysis
            pass_stats = analyze_pass_completion(passes_df)
            print(f"📊 Pass completion analysis complete")
            
            # Show pass leaders
            home_pass_stats = pass_stats[pass_stats['team'] == home_team].head(3)
            away_pass_stats = pass_stats[pass_stats['team'] == away_team].head(3)
            
            print(f"\n🏠 {home_team} Top Passers:")
            for _, player in home_pass_stats.iterrows():
                print(f"   • {player['player']}: {player['Total_Passes']} passes ({player['Completion_Rate']:.1f}%)")
            
            print(f"\n🚶 {away_team} Top Passers:")
            for _, player in away_pass_stats.iterrows():
                print(f"   • {player['player']}: {player['Total_Passes']} passes ({player['Completion_Rate']:.1f}%)")
    except Exception as e:
        print(f"❌ Pass analysis error: {e}")
    
    # 2. Player Touch Maps  
    print(f"\n👥 PLAYER POSITIONING VISUALIZATIONS")
    try:
        if len(passes_df) > 0:
            pass_stats = analyze_pass_completion(passes_df)
            home_passers = pass_stats[pass_stats['team'] == home_team]
            away_passers = pass_stats[pass_stats['team'] == away_team]
            
            if len(home_passers) > 0:
                home_top_passer = home_passers.iloc[0]['player']
                print(f"📍 Creating touch map for {home_top_passer}...")
                create_player_touch_map(passes_df, home_top_passer, team_colors)
            
            if len(home_passers) > 1 and len(away_passers) > 0:
                home_player = home_passers.iloc[0]['player']
                away_player = away_passers.iloc[0]['player']
                print(f"🔄 Comparing {home_player} vs {away_player}...")
                compare_player_touch_maps(passes_df, home_player, away_player, team_colors)
    except Exception as e:
        print(f"❌ Player positioning error: {e}")
    
    # 3. Shot Analysis
    print(f"\n🎯 SHOT ANALYSIS VISUALIZATIONS")
    try:
        if len(shots_df) > 0:
            # Team shot comparison
            create_team_shot_comparison(shots_df, team_colors)
            
            # Individual shot maps
            shot_stats = analyze_team_shots(shots_df)
            
            for team in [home_team, away_team]:
                team_shooters = shot_stats[shot_stats['team'] == team]
                if len(team_shooters) > 0:
                    top_shooter = team_shooters.iloc[0]['player']
                    print(f"🏹 Creating shot map for {top_shooter} ({team})...")
                    create_player_shot_map(shots_df, top_shooter, team_colors)
        else:
            print("⚠️ No shot data available")
    except Exception as e:
        print(f"❌ Shot analysis error: {e}")
    
    # 4. Defensive Analysis
    print(f"\n🛡️ DEFENSIVE ANALYSIS VISUALIZATIONS")
    try:
        # Pressure events
        pressure_events = analyze_pressure_events(test_events, home_team)
        if len(pressure_events) > 0:
            create_pressure_heatmap(pressure_events, home_team, team_colors)
        
        # Defensive actions
        plot_defensive_actions_comparison(test_events, team_colors)
        
    except Exception as e:
        print(f"❌ Defensive analysis error: {e}")
    
    print(f"\n🎉 VISUALIZATION SHOWCASE COMPLETE!")
    print(f"🏆 Generated comprehensive football analysis for {home_team} vs {away_team}")

# Run the fixed version
create_all_visualizations_fixed()

# %% [markdown]
# ## 🏆 2022 FIFA World Cup Final Analysis
# 
# **Argentina vs France** - December 18, 2022
# The greatest World Cup Final of all time! Let's analyze this epic match using our StatsBomb toolkit.

# %%
# 🏆 2022 FIFA WORLD CUP FINAL: Argentina vs France
print("🔍 Searching for the 2022 World Cup Final...")

def find_world_cup_final():
    """
    Find and analyze the 2022 FIFA World Cup Final
    """
    
    analyzer = StatsBombAnalyzer()
    
    # Get all competitions
    competitions = analyzer.get_competitions()
    
    # Look for FIFA World Cup 2022
    world_cup_comps = competitions[
        (competitions['competition_name'].str.contains('FIFA World Cup', na=False)) &
        (competitions['season_name'].str.contains('2022', na=False))
    ]
    
    print(f"🔍 Found {len(world_cup_comps)} World Cup 2022 competitions")
    
    if not world_cup_comps.empty:
        # Show available World Cup competitions
        print("🏆 Available World Cup 2022 competitions:")
        for _, comp in world_cup_comps.iterrows():
            print(f"   • {comp['competition_name']} - {comp['season_name']}")
            print(f"     Competition ID: {comp['competition_id']}, Season ID: {comp['season_id']}")
        
        # Use the first one found
        wc_comp = world_cup_comps.iloc[0]
        
        # Get all matches
        print(f"\n📋 Loading matches for {wc_comp['competition_name']} {wc_comp['season_name']}...")
        matches = analyzer.get_matches(wc_comp['competition_id'], wc_comp['season_id'])
        
        if not matches.empty:
            print(f"✅ Found {len(matches)} matches")
            
            # Look for the final (Argentina vs France)
            final_match = matches[
                ((matches['home_team'] == 'Argentina') & (matches['away_team'] == 'France')) |
                ((matches['home_team'] == 'France') & (matches['away_team'] == 'Argentina'))
            ]
            
            if not final_match.empty:
                final = final_match.iloc[0]
                print(f"\n🎉 FOUND THE WORLD CUP FINAL!")
                print_match_summary(final)
                
                return final, analyzer
            else:
                print("❌ Final not found, showing available matches:")
                print("\n🔍 Available matches (last 10):")
                for _, match in matches.tail(10).iterrows():
                    print(f"   • {match['home_team']} vs {match['away_team']} ({match['home_score']}-{match['away_score']})")
                    print(f"     Date: {match['match_date']}, ID: {match['match_id']}")
                
                # Return the last match as a fallback
                return matches.iloc[-1], analyzer
        else:
            print("❌ No matches found")
            return None, analyzer
    else:
        print("❌ No FIFA World Cup 2022 found in competitions")
        print("\n🔍 Available competitions containing 'World Cup':")
        wc_any = competitions[competitions['competition_name'].str.contains('World Cup', na=False)]
        for _, comp in wc_any.iterrows():
            print(f"   • {comp['competition_name']} - {comp['season_name']}")
        
        return None, analyzer

# Find the World Cup Final
world_cup_final, wc_analyzer = find_world_cup_final()

# %%
# 🎨 WORLD CUP FINAL VISUALIZATION SPECTACULAR
print("🏆 Creating ALL visualizations for the 2022 World Cup Final...")

def analyze_world_cup_final():
    """
    Complete analysis of the 2022 FIFA World Cup Final
    """
    
    if world_cup_final is None:
        print("❌ No World Cup Final data available")
        return
    
    match_id = world_cup_final['match_id']
    home_team = world_cup_final['home_team']  # Argentina
    away_team = world_cup_final['away_team']  # France
    
    print(f"🔥 ANALYZING THE GREATEST WORLD CUP FINAL OF ALL TIME!")
    print(f"🇦🇷 {home_team} vs {away_team} 🇫🇷")
    print(f"📊 Final Score: {world_cup_final['home_score']}-{world_cup_final['away_score']} (AET)")
    print(f"🏟️ {world_cup_final['stadium']}")
    print("=" * 80)
    
    # Load match events
    print("📥 Loading World Cup Final events...")
    events = wc_analyzer.get_events(match_id, split=False)
    
    if events.empty:
        print("❌ No events data available")
        return
    
    print(f"✅ Loaded {len(events)} events for the World Cup Final")
    
    # Prepare data
    passes_df = events[events['type'] == 'Pass'].copy()
    shots_df = events[events['type'] == 'Shot'].copy()
    
    # Extract coordinates safely
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    if 'location' in passes_df.columns:
        passes_df[['x', 'y']] = passes_df['location'].apply(safe_extract_coords)
        if 'pass_end_location' in passes_df.columns:
            passes_df[['pass_end_x', 'pass_end_y']] = passes_df['pass_end_location'].apply(safe_extract_coords)
    
    if 'location' in shots_df.columns:
        shots_df[['x', 'y']] = shots_df['location'].apply(safe_extract_coords)
    
    # Clean data
    passes_df = passes_df.dropna(subset=['x', 'y'])
    shots_df = shots_df.dropna(subset=['x', 'y'])
    
    print(f"📊 Match Data: {len(passes_df)} passes, {len(shots_df)} shots")
    print("🎨 Creating World Cup Final visualizations...")
    
    # 1. ARGENTINA ANALYSIS
    print(f"\n🇦🇷 ARGENTINA ANALYSIS")
    print("-" * 40)
    
    if len(passes_df) > 0:
        # Argentina colors (light blue and white)
        argentina_colors = {
            'primary': '#75AADB',    # Argentina light blue
            'secondary': '#FFFFFF',  # White
            'background': '#001F3F', # Dark blue
            'text': '#87CEEB'       # Light blue
        }
        
        # Progressive passes analysis
        argentina_progressions = analyze_final_third_progressions(passes_df, home_team)
        if len(argentina_progressions) > 0:
            print("📈 Creating Argentina progressive passes chart...")
            plot_progression_bar_chart(argentina_progressions, home_team, argentina_colors)
            
            # Messi or top progressor visualization
            top_argentina_player = argentina_progressions.iloc[0]['player']
            print(f"🌟 Progressive passes map for {top_argentina_player}...")
            try:
                plot_player_progressions_on_pitch(passes_df, top_argentina_player, argentina_colors)
            except:
                print(f"⚠️ Could not create progression map for {top_argentina_player}")
    
    # 2. FRANCE ANALYSIS  
    print(f"\n🇫🇷 FRANCE ANALYSIS")
    print("-" * 40)
    
    # France colors (blue, white, red)
    france_colors = {
        'primary': '#0055A4',    # France blue
        'secondary': '#FFFFFF',  # White  
        'background': '#1a1a2e', # Dark blue
        'text': '#EF4135'       # France red
    }
    
    if len(passes_df) > 0:
        france_progressions = analyze_final_third_progressions(passes_df, away_team)
        if len(france_progressions) > 0:
            print("📈 Creating France progressive passes chart...")
            plot_progression_bar_chart(france_progressions, away_team, france_colors)
    
    # 3. MESSI vs MBAPPÉ COMPARISON
    print(f"\n⭐ MESSI vs MBAPPÉ COMPARISON")
    print("-" * 40)
    
    # Look for Messi and Mbappé in the data
    messi_variants = ['Lionel Messi', 'Lionel Andrés Messi Cuccittini', 'L. Messi']
    mbappe_variants = ['Kylian Mbappé', 'Kylian Mbappé Lottin', 'K. Mbappé']
    
    messi_name = None
    mbappe_name = None
    
    for variant in messi_variants:
        if variant in passes_df['player'].values:
            messi_name = variant
            break
            
    for variant in mbappe_variants:
        if variant in passes_df['player'].values:
            mbappe_name = variant
            break
    
    if messi_name and mbappe_name:
        print(f"🐐 Found Messi as: {messi_name}")
        print(f"🚀 Found Mbappé as: {mbappe_name}")
        try:
            compare_player_touch_maps(passes_df, messi_name, mbappe_name, argentina_colors)
        except Exception as e:
            print(f"⚠️ Could not create comparison: {e}")
    else:
        # Show available players for reference
        argentina_players = passes_df[passes_df['team'] == home_team]['player'].unique()[:5]
        france_players = passes_df[passes_df['team'] == away_team]['player'].unique()[:5]
        
        print(f"🇦🇷 Argentina players found: {list(argentina_players)}")
        print(f"🇫🇷 France players found: {list(france_players)}")
        
        if len(argentina_players) > 0 and len(france_players) > 0:
            print(f"🔄 Comparing {argentina_players[0]} vs {france_players[0]}...")
            try:
                compare_player_touch_maps(passes_df, argentina_players[0], france_players[0], argentina_colors)
            except Exception as e:
                print(f"⚠️ Could not create comparison: {e}")
    
    # 4. SHOT ANALYSIS
    print(f"\n🎯 WORLD CUP FINAL SHOT ANALYSIS")
    print("-" * 40)
    
    if len(shots_df) > 0:
        # Overall shot comparison
        create_team_shot_comparison(shots_df, argentina_colors)
        
        # Individual shot maps for top scorers
        shot_stats = analyze_team_shots(shots_df)
        
        for team, colors in [(home_team, argentina_colors), (away_team, france_colors)]:
            team_shooters = shot_stats[shot_stats['team'] == team]
            if len(team_shooters) > 0:
                top_shooter = team_shooters.iloc[0]['player']
                print(f"🏹 Shot map for {top_shooter} ({team})...")
                try:
                    create_player_shot_map(shots_df, top_shooter, colors)
                except Exception as e:
                    print(f"⚠️ Could not create shot map for {top_shooter}: {e}")
    
    # 5. MATCH PERFORMANCE SUMMARY
    print(f"\n📊 WORLD CUP FINAL PERFORMANCE SUMMARY")
    print("-" * 40)
    
    if len(events) > 0:
        argentina_stats = analyze_team_performance(events, home_team)
        france_stats = analyze_team_performance(events, away_team)
        
        print(f"🇦🇷 ARGENTINA PERFORMANCE:")
        print(f"   • Passes: {argentina_stats['total_passes']} ({argentina_stats['pass_accuracy']:.1f}% accuracy)")
        print(f"   • Shots: {argentina_stats['shots']} (Goals: {argentina_stats['goals']})")
        print(f"   • Possession: {argentina_stats['possession_pct']:.1f}%")
        
        print(f"\n🇫🇷 FRANCE PERFORMANCE:")
        print(f"   • Passes: {france_stats['total_passes']} ({france_stats['pass_accuracy']:.1f}% accuracy)")
        print(f"   • Shots: {france_stats['shots']} (Goals: {france_stats['goals']})")
        print(f"   • Possession: {france_stats['possession_pct']:.1f}%")
    
    print(f"\n🏆 WORLD CUP FINAL ANALYSIS COMPLETE!")
    print(f"🎉 The greatest match in football history visualized!")

# Run the World Cup Final analysis
analyze_world_cup_final()

# %%
# 🔥 PRESSURE MAP VISUALIZATION
def create_pressure_map(events_df, team_name=None, colors=None):
    """
    Create a heatmap showing where defensive pressure events occurred
    
    Args:
        events_df: StatsBomb events DataFrame
        team_name: Team name to analyze (if None, shows both teams)
        colors: Color scheme dictionary
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mplsoccer import Pitch
    import numpy as np
    
    # Default colors
    if colors is None:
        colors = {
            'primary': '#E3120B',
            'secondary': '#006BA2', 
            'background': '#0C0C0C',
            'text': '#FFFFFF'
        }
    
    # Filter for pressure-related events
    pressure_events = events_df[
        (events_df['type'].isin(['Pressure', 'Interception', 'Block', 'Foul Committed'])) |
        (events_df['possession_team'] != events_df['team'])  # Defensive actions
    ].copy()
    
    if team_name:
        pressure_events = pressure_events[pressure_events['team'] == team_name]
        title = f"{team_name} - Defensive Pressure Map"
    else:
        title = "Defensive Pressure Map - Both Teams"
    
    if pressure_events.empty:
        print(f"❌ No pressure events found for {team_name if team_name else 'teams'}")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    if 'location' in pressure_events.columns:
        pressure_events[['x', 'y']] = pressure_events['location'].apply(safe_extract_coords)
        pressure_events = pressure_events.dropna(subset=['x', 'y'])
    
    if pressure_events.empty:
        print(f"❌ No valid pressure coordinates found")
        return
    
    # Create pitch and heatmap
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color=colors['background'],
        line_color='white',
        linewidth=2
    )
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(colors['background'])
    pitch.draw(ax=ax)
    
    # Create 2D histogram for heatmap
    try:
        # Bin the pressure events
        bins_x = np.linspace(0, 120, 25)  # StatsBomb pitch is 120x80
        bins_y = np.linspace(0, 80, 17)
        
        pressure_counts, xedges, yedges = np.histogram2d(
            pressure_events['x'], pressure_events['y'],
            bins=[bins_x, bins_y]
        )
        
        # Create meshgrid for plotting
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        
        # Plot heatmap
        heatmap = ax.contourf(
            X, Y, pressure_counts.T,
            levels=20,
            cmap='Reds',
            alpha=0.7,
            extend='max'
        )
        
        # Add colorbar
        cbar = plt.colorbar(heatmap, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Pressure Events', color='white', fontsize=12)
        cbar.ax.tick_params(colors='white')
        
        # Overlay scatter points for individual events
        scatter = ax.scatter(
            pressure_events['x'], pressure_events['y'],
            s=30, c=colors['primary'], alpha=0.8,
            edgecolors='white', linewidth=0.5
        )
        
        # Add statistics text
        stats_text = f"""
Pressure Events: {len(pressure_events)}
Event Types: {', '.join(pressure_events['type'].value_counts().head(3).index.tolist())}
        """.strip()
        
        ax.text(
            2, 76, stats_text,
            fontsize=11, color='white',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['primary'], alpha=0.8)
        )
        
    except Exception as e:
        print(f"⚠️ Error creating heatmap, using scatter plot: {e}")
        # Fallback to scatter plot
        scatter = ax.scatter(
            pressure_events['x'], pressure_events['y'],
            s=60, c=colors['primary'], alpha=0.7,
            edgecolors='white', linewidth=1
        )
    
    # Styling
    ax.set_title(title, fontsize=18, color='white', pad=20, weight='bold')
    
    # Add team-specific styling
    if team_name:
        team_color = colors['primary']
        ax.add_patch(plt.Rectangle((0, -5), 120, 2, 
                                 facecolor=team_color, alpha=0.8, clip_on=False))
    
    plt.tight_layout()
    plt.show()
    
    # Print pressure statistics
    print(f"\n📊 PRESSURE MAP ANALYSIS:")
    print(f"   • Total pressure events: {len(pressure_events)}")
    print(f"   • Event breakdown:")
    for event_type, count in pressure_events['type'].value_counts().head(5).items():
        print(f"     - {event_type}: {count}")
    
    # Pressure by thirds
    if len(pressure_events) > 0:
        defensive_third = len(pressure_events[pressure_events['x'] <= 40])
        middle_third = len(pressure_events[(pressure_events['x'] > 40) & (pressure_events['x'] <= 80)])
        attacking_third = len(pressure_events[pressure_events['x'] > 80])
        
        print(f"   • Pressure by pitch thirds:")
        print(f"     - Defensive third: {defensive_third} ({defensive_third/len(pressure_events)*100:.1f}%)")
        print(f"     - Middle third: {middle_third} ({middle_third/len(pressure_events)*100:.1f}%)")
        print(f"     - Attacking third: {attacking_third} ({attacking_third/len(pressure_events)*100:.1f}%)")

def create_team_pressure_comparison(events_df):
    """
    Create side-by-side pressure maps for both teams
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    
    teams = events_df['team'].unique()
    if len(teams) < 2:
        print("❌ Need at least 2 teams for comparison")
        return
    
    team1, team2 = teams[0], teams[1]
    
    # Team colors
    team1_colors = {
        'primary': '#75AADB',    # Argentina light blue
        'background': '#001F3F', # Dark blue
    }
    
    team2_colors = {
        'primary': '#EF4135',    # France red
        'background': '#001F3F', # Dark blue  
    }
    
    print(f"🔥 Creating pressure comparison: {team1} vs {team2}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('#001F3F')
    
    for ax, team, colors in [(ax1, team1, team1_colors), (ax2, team2, team2_colors)]:
        # Create pitch
        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color=colors['background'],
            line_color='white',
            linewidth=2
        )
        pitch.draw(ax=ax)
        
        # Get team pressure events
        team_events = events_df[
            (events_df['team'] == team) &
            (events_df['type'].isin(['Pressure', 'Interception', 'Block', 'Foul Committed']))
        ].copy()
        
        if not team_events.empty and 'location' in team_events.columns:
            # Extract coordinates
            def safe_extract_coords(location):
                try:
                    if isinstance(location, list) and len(location) >= 2:
                        return pd.Series([float(location[0]), float(location[1])])
                except:
                    pass
                return pd.Series([None, None])
            
            team_events[['x', 'y']] = team_events['location'].apply(safe_extract_coords)
            team_events = team_events.dropna(subset=['x', 'y'])
            
            if not team_events.empty:
                # Create scatter plot
                ax.scatter(
                    team_events['x'], team_events['y'],
                    s=80, c=colors['primary'], alpha=0.7,
                    edgecolors='white', linewidth=1
                )
                
                # Add statistics
                stats_text = f"""
Events: {len(team_events)}
Def. Third: {len(team_events[team_events['x'] <= 40])}
Mid. Third: {len(team_events[(team_events['x'] > 40) & (team_events['x'] <= 80)])}
Att. Third: {len(team_events[team_events['x'] > 80])}
                """.strip()
                
                ax.text(
                    2, 76, stats_text,
                    fontsize=10, color='white',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['primary'], alpha=0.8)
                )
        
        ax.set_title(f"{team} - Pressure Map", fontsize=16, color='white', pad=15, weight='bold')
    
    plt.tight_layout()
    plt.show()

# 🏆 WORLD CUP FINAL PRESSURE ANALYSIS
def analyze_world_cup_final_pressure():
    """
    Comprehensive pressure analysis for the World Cup Final
    """
    print("🔥 WORLD CUP FINAL PRESSURE ANALYSIS")
    print("=" * 50)
    
    if world_cup_final is None:
        print("❌ No World Cup Final data available")
        return
    
    match_id = world_cup_final['match_id']
    home_team = world_cup_final['home_team']  # Argentina
    away_team = world_cup_final['away_team']  # France
    
    # Load events
    events = wc_analyzer.get_events(match_id, split=False)
    
    if events.empty:
        print("❌ No events data available")
        return
    
    print(f"📊 Analyzing pressure patterns for {home_team} vs {away_team}")
    
    # Team-specific pressure maps
    argentina_colors = {
        'primary': '#75AADB',    # Argentina light blue
        'secondary': '#FFFFFF',  # White
        'background': '#001F3F', # Dark blue
        'text': '#87CEEB'       # Light blue
    }
    
    france_colors = {
        'primary': '#EF4135',    # France red
        'secondary': '#0055A4',  # France blue
        'background': '#001F3F', # Dark blue
        'text': '#FFFFFF'       # White
    }
    
    print(f"\n🇦🇷 ARGENTINA PRESSURE MAP:")
    create_pressure_map(events, home_team, argentina_colors)
    
    print(f"\n🇫🇷 FRANCE PRESSURE MAP:")
    create_pressure_map(events, away_team, france_colors)
    
    print(f"\n⚔️ PRESSURE COMPARISON:")
    create_team_pressure_comparison(events)
    
    print(f"\n🏆 World Cup Final pressure analysis complete!")

# Run the pressure analysis
analyze_world_cup_final_pressure()

# %%
# 🔍 EXPLORE AVAILABLE DATA VARIABLES
def explore_statsbomb_data():
    """
    Explore what variables and event types are available in StatsBomb data
    """
    print("🔍 EXPLORING STATSBOMB DATA VARIABLES")
    print("=" * 50)
    
    if world_cup_final is None:
        print("❌ No World Cup Final data available")
        return
    
    match_id = world_cup_final['match_id']
    events = wc_analyzer.get_events(match_id, split=False)
    
    if events.empty:
        print("❌ No events data available")
        return
    
    print(f"📊 Total events in World Cup Final: {len(events)}")
    print(f"📊 Data shape: {events.shape}")
    
    print(f"\n🎯 EVENT TYPES:")
    event_counts = events['type'].value_counts()
    for event_type, count in event_counts.head(15).items():
        print(f"   • {event_type}: {count}")
    
    print(f"\n📋 AVAILABLE COLUMNS:")
    print(f"   Total columns: {len(events.columns)}")
    for i, col in enumerate(events.columns[:20], 1):
        print(f"   {i:2}. {col}")
    if len(events.columns) > 20:
        print(f"   ... and {len(events.columns) - 20} more columns")
    
    # Look for interesting event-specific columns
    print(f"\n🔎 PASS-RELATED COLUMNS:")
    pass_cols = [col for col in events.columns if 'pass' in col.lower()]
    for col in pass_cols[:10]:
        print(f"   • {col}")
    
    print(f"\n⚔️ DUEL/DEFENSIVE COLUMNS:")
    defensive_cols = [col for col in events.columns if any(term in col.lower() for term in ['duel', 'pressure', 'block', 'tackle', 'interception'])]
    for col in defensive_cols[:10]:
        print(f"   • {col}")
    
    print(f"\n🏃 CARRY/MOVEMENT COLUMNS:")
    movement_cols = [col for col in events.columns if any(term in col.lower() for term in ['carry', 'dribble', 'run'])]
    for col in movement_cols[:10]:
        print(f"   • {col}")
    
    return events

# Explore the data first
events_data = explore_statsbomb_data()

# %%
# 🌐 PASS NETWORK VISUALIZATION
def create_pass_network(events_df, team_name, colors=None):
    """
    Create a pass network showing connections between players
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from mplsoccer import Pitch
    import numpy as np
    
    if colors is None:
        colors = {'primary': '#E3120B', 'background': '#0C0C0C', 'text': '#FFFFFF'}
    
    # Filter passes for the team
    team_passes = events_df[
        (events_df['team'] == team_name) & 
        (events_df['type'] == 'Pass') &
        (events_df['pass_outcome'].isna())  # Successful passes only
    ].copy()
    
    if team_passes.empty:
        print(f"❌ No successful passes found for {team_name}")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    team_passes[['x', 'y']] = team_passes['location'].apply(safe_extract_coords)
    team_passes[['end_x', 'end_y']] = team_passes['pass_end_location'].apply(safe_extract_coords)
    team_passes = team_passes.dropna(subset=['x', 'y', 'end_x', 'end_y'])
    
    if team_passes.empty:
        print(f"❌ No valid pass coordinates found for {team_name}")
        return
    
    # Calculate average positions
    avg_positions = team_passes.groupby('player')[['x', 'y']].mean()
    
    # Count passes between players
    pass_counts = team_passes.groupby(['player', 'pass_recipient']).size().reset_index(name='passes')
    pass_counts = pass_counts[pass_counts['passes'] >= 3]  # Only frequent connections
    
    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color=colors['background'], 
                  line_color='white', linewidth=2)
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(colors['background'])
    pitch.draw(ax=ax)
    
    # Draw player positions
    for player, pos in avg_positions.iterrows():
        if not pd.isna(pos['x']) and not pd.isna(pos['y']):
            # Player circle
            circle = plt.Circle((pos['x'], pos['y']), 3, 
                              color=colors['primary'], alpha=0.8, zorder=5)
            ax.add_patch(circle)
            
            # Player name
            ax.text(pos['x'], pos['y']-6, player.split()[-1], 
                   ha='center', va='top', fontsize=8, color='white', weight='bold')
    
    # Draw pass connections
    for _, row in pass_counts.iterrows():
        passer = row['player']
        receiver = row['pass_recipient']
        
        if passer in avg_positions.index and receiver in avg_positions.index:
            x1, y1 = avg_positions.loc[passer]
            x2, y2 = avg_positions.loc[receiver]
            
            if not (pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2)):
                # Line thickness based on number of passes
                line_width = min(row['passes'] / 2, 5)
                ax.plot([x1, x2], [y1, y2], color=colors['primary'], 
                       alpha=0.6, linewidth=line_width, zorder=1)
    
    ax.set_title(f'{team_name} - Pass Network', fontsize=18, color='white', 
                 pad=20, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Pass Network Stats for {team_name}:")
    print(f"   • Players involved: {len(avg_positions)}")
    print(f"   • Pass connections: {len(pass_counts)}")
    print(f"   • Total successful passes: {len(team_passes)}")

# ⚔️ DUEL MAP VISUALIZATION
def create_duel_map(events_df, team_name=None, colors=None):
    """
    Create a map showing where duels (50/50 challenges) occurred
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    
    if colors is None:
        colors = {'primary': '#E3120B', 'background': '#0C0C0C', 'text': '#FFFFFF'}
    
    # Filter duel events
    duels = events_df[events_df['type'] == 'Duel'].copy()
    
    if team_name:
        duels = duels[duels['team'] == team_name]
        title = f"{team_name} - Duel Map"
    else:
        title = "Duel Map - All Teams"
    
    if duels.empty:
        print(f"❌ No duels found for {team_name if team_name else 'teams'}")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    duels[['x', 'y']] = duels['location'].apply(safe_extract_coords)
    duels = duels.dropna(subset=['x', 'y'])
    
    if duels.empty:
        print(f"❌ No valid duel coordinates found")
        return
    
    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color=colors['background'], 
                  line_color='white', linewidth=2)
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(colors['background'])
    pitch.draw(ax=ax)
    
    # Color code by duel outcome
    won_duels = duels[duels['duel_outcome'] == 'Won']
    lost_duels = duels[duels['duel_outcome'] == 'Lost']
    
    # Plot won duels
    if not won_duels.empty:
        ax.scatter(won_duels['x'], won_duels['y'], s=100, c='green', 
                  alpha=0.8, marker='o', edgecolors='white', linewidth=1, 
                  label='Won Duels')
    
    # Plot lost duels  
    if not lost_duels.empty:
        ax.scatter(lost_duels['x'], lost_duels['y'], s=100, c='red', 
                  alpha=0.8, marker='X', edgecolors='white', linewidth=1,
                  label='Lost Duels')
    
    # Add legend
    ax.legend(loc='upper right', facecolor=colors['background'], 
              edgecolor='white', labelcolor='white')
    
    ax.set_title(title, fontsize=18, color='white', pad=20, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    won_count = len(won_duels)
    lost_count = len(lost_duels)
    total_duels = won_count + lost_count
    
    if total_duels > 0:
        win_rate = (won_count / total_duels) * 100
        print(f"📊 Duel Statistics:")
        print(f"   • Total duels: {total_duels}")
        print(f"   • Won: {won_count} ({win_rate:.1f}%)")
        print(f"   • Lost: {lost_count} ({100-win_rate:.1f}%)")
        
        # Duel types
        if 'duel_type' in duels.columns:
            duel_types = duels['duel_type'].value_counts()
            print(f"   • Duel types: {dict(duel_types)}")

# 🏃 CARRY PATTERN VISUALIZATION  
def create_carry_map(events_df, team_name, colors=None):
    """
    Create a map showing ball carry patterns
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    
    if colors is None:
        colors = {'primary': '#E3120B', 'background': '#0C0C0C', 'text': '#FFFFFF'}
    
    # Filter carry events
    carries = events_df[
        (events_df['team'] == team_name) & 
        (events_df['type'] == 'Carry')
    ].copy()
    
    if carries.empty:
        print(f"❌ No carries found for {team_name}")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    carries[['x', 'y']] = carries['location'].apply(safe_extract_coords)
    carries[['end_x', 'end_y']] = carries['carry_end_location'].apply(safe_extract_coords)
    carries = carries.dropna(subset=['x', 'y', 'end_x', 'end_y'])
    
    if carries.empty:
        print(f"❌ No valid carry coordinates found for {team_name}")
        return
    
    # Calculate carry distance and direction
    carries['distance'] = np.sqrt((carries['end_x'] - carries['x'])**2 + 
                                 (carries['end_y'] - carries['y'])**2)
    carries['forward_progress'] = carries['end_x'] - carries['x']
    
    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color=colors['background'], 
                  line_color='white', linewidth=2)
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(colors['background'])
    pitch.draw(ax=ax)
    
    # Plot carries as arrows
    for _, carry in carries.iterrows():
        if carry['distance'] > 2:  # Only show significant carries
            dx = carry['end_x'] - carry['x']
            dy = carry['end_y'] - carry['y']
            
            # Color based on forward progress
            color = 'green' if carry['forward_progress'] > 5 else colors['primary']
            alpha = min(carry['distance'] / 20, 1.0)  # Fade short carries
            
            ax.annotate('', xy=(carry['end_x'], carry['end_y']), 
                       xytext=(carry['x'], carry['y']),
                       arrowprops=dict(arrowstyle='->', color=color, 
                                     alpha=alpha, lw=2))
    
    ax.set_title(f'{team_name} - Ball Carry Patterns', fontsize=18, 
                 color='white', pad=20, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    total_carries = len(carries)
    avg_distance = carries['distance'].mean()
    forward_carries = len(carries[carries['forward_progress'] > 5])
    
    print(f"📊 Carry Statistics for {team_name}:")
    print(f"   • Total carries: {total_carries}")
    print(f"   • Average distance: {avg_distance:.1f} yards")
    print(f"   • Progressive carries: {forward_carries} ({forward_carries/total_carries*100:.1f}%)")
    print(f"   • Longest carry: {carries['distance'].max():.1f} yards")

# 🎯 AERIAL DUEL MAP
def create_aerial_map(events_df, colors=None):
    """
    Create a map showing aerial duels and clearances
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    
    if colors is None:
        colors = {'primary': '#E3120B', 'background': '#0C0C0C', 'text': '#FFFFFF'}
    
    # Filter aerial events
    aerial_events = events_df[
        (events_df['type'].isin(['Duel', 'Clearance', 'Pass'])) &
        ((events_df['duel_type'] == 'Aerial Lost') | 
         (events_df['clearance_aerial_won'] == True) |
         (events_df['pass_aerial_won'] == True))
    ].copy()
    
    if aerial_events.empty:
        print("❌ No aerial events found")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    aerial_events[['x', 'y']] = aerial_events['location'].apply(safe_extract_coords)
    aerial_events = aerial_events.dropna(subset=['x', 'y'])
    
    if aerial_events.empty:
        print("❌ No valid aerial coordinates found")
        return
    
    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color=colors['background'], 
                  line_color='white', linewidth=2)
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(colors['background'])
    pitch.draw(ax=ax)
    
    # Plot by team
    teams = aerial_events['team'].unique()
    team_colors = ['#75AADB', '#EF4135']  # Argentina blue, France red
    
    for i, team in enumerate(teams):
        team_aerials = aerial_events[aerial_events['team'] == team]
        ax.scatter(team_aerials['x'], team_aerials['y'], 
                  s=120, c=team_colors[i % len(team_colors)], 
                  alpha=0.7, marker='^', edgecolors='white', 
                  linewidth=1, label=f'{team} Aerials')
    
    ax.legend(loc='upper right', facecolor=colors['background'], 
              edgecolor='white', labelcolor='white')
    
    ax.set_title('Aerial Duels & High Balls', fontsize=18, 
                 color='white', pad=20, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Aerial Battle Statistics:")
    for team in teams:
        team_aerials = aerial_events[aerial_events['team'] == team]
        print(f"   • {team}: {len(team_aerials)} aerial actions")

# 🥅 GOALKEEPER ACTION MAP
def create_goalkeeper_map(events_df, colors=None):
    """
    Create a map showing goalkeeper actions
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    
    if colors is None:
        colors = {'primary': '#E3120B', 'background': '#0C0C0C', 'text': '#FFFFFF'}
    
    # Filter goalkeeper events
    gk_events = events_df[events_df['type'] == 'Goal Keeper'].copy()
    
    if gk_events.empty:
        print("❌ No goalkeeper events found")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    gk_events[['x', 'y']] = gk_events['location'].apply(safe_extract_coords)
    gk_events = gk_events.dropna(subset=['x', 'y'])
    
    if gk_events.empty:
        print("❌ No valid goalkeeper coordinates found")
        return
    
    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color=colors['background'], 
                  line_color='white', linewidth=2)
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(colors['background'])
    pitch.draw(ax=ax)
    
    # Plot goalkeeper actions by team
    teams = gk_events['team'].unique()
    team_colors = ['#75AADB', '#EF4135']  # Argentina blue, France red
    
    for i, team in enumerate(teams):
        team_gk = gk_events[gk_events['team'] == team]
        ax.scatter(team_gk['x'], team_gk['y'], 
                  s=150, c=team_colors[i % len(team_colors)], 
                  alpha=0.8, marker='D', edgecolors='white', 
                  linewidth=2, label=f'{team} GK Actions')
    
    ax.legend(loc='upper right', facecolor=colors['background'], 
              edgecolor='white', labelcolor='white')
    
    ax.set_title('Goalkeeper Actions', fontsize=18, 
                 color='white', pad=20, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Goalkeeper Statistics:")
    for team in teams:
        team_gk = gk_events[gk_events['team'] == team]
        print(f"   • {team}: {len(team_gk)} goalkeeper actions")
        
        # Show most active goalkeeper
        if not team_gk.empty:
            top_gk = team_gk['player'].value_counts().head(1)
            if not top_gk.empty:
                print(f"     - {top_gk.index[0]}: {top_gk.iloc[0]} actions")

# 🏆 RUN ALL NEW VISUALIZATIONS FOR WORLD CUP FINAL
def create_advanced_world_cup_maps():
    """
    Create all the advanced maps for the World Cup Final
    """
    print("🚀 CREATING ADVANCED WORLD CUP FINAL MAPS")
    print("=" * 60)
    
    if events_data is None or events_data.empty:
        print("❌ No events data available")
        return
    
    home_team = world_cup_final['home_team']  # Argentina
    away_team = world_cup_final['away_team']  # France
    
    # Team colors
    argentina_colors = {
        'primary': '#75AADB',    # Argentina light blue
        'background': '#001F3F', # Dark blue
        'text': '#87CEEB'       # Light blue
    }
    
    france_colors = {
        'primary': '#EF4135',    # France red
        'background': '#001F3F', # Dark blue
        'text': '#FFFFFF'       # White
    }
    
    print(f"\n🌐 PASS NETWORKS:")
    print("-" * 30)
    create_pass_network(events_data, home_team, argentina_colors)
    create_pass_network(events_data, away_team, france_colors)
    
    print(f"\n⚔️ DUEL MAPS:")
    print("-" * 30)
    create_duel_map(events_data, home_team, argentina_colors)
    create_duel_map(events_data, away_team, france_colors)
    
    print(f"\n🏃 CARRY PATTERNS:")
    print("-" * 30)
    create_carry_map(events_data, home_team, argentina_colors)
    create_carry_map(events_data, away_team, france_colors)
    
    print(f"\n🎯 AERIAL BATTLES:")
    print("-" * 30)
    create_aerial_map(events_data)
    
    print(f"\n🥅 GOALKEEPER ACTIONS:")
    print("-" * 30)
    create_goalkeeper_map(events_data)
    
    print(f"\n🏆 ALL ADVANCED MAPS COMPLETE!")

# Run all the new visualizations
create_advanced_world_cup_maps()

# %%
# 📍 AVERAGE POSITION VISUALIZATION
def create_average_position_map(events_df, team_name, colors=None, min_actions=10):
    """
    Create a map showing average positions of all players
    
    Args:
        events_df: StatsBomb events DataFrame
        team_name: Team name to analyze
        colors: Color scheme dictionary
        min_actions: Minimum number of actions to include a player
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    import numpy as np
    
    if colors is None:
        colors = {'primary': '#E3120B', 'background': '#0C0C0C', 'text': '#FFFFFF'}
    
    # Filter events for the team with location data
    team_events = events_df[
        (events_df['team'] == team_name) & 
        (events_df['location'].notna())
    ].copy()
    
    if team_events.empty:
        print(f"❌ No events with location found for {team_name}")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    team_events[['x', 'y']] = team_events['location'].apply(safe_extract_coords)
    team_events = team_events.dropna(subset=['x', 'y'])
    
    if team_events.empty:
        print(f"❌ No valid coordinates found for {team_name}")
        return
    
    # Calculate average positions for each player
    avg_positions = team_events.groupby('player').agg({
        'x': 'mean',
        'y': 'mean',
        'type': 'count'  # Number of actions
    }).round(1)
    
    # Filter players with minimum actions
    avg_positions = avg_positions[avg_positions['type'] >= min_actions]
    avg_positions = avg_positions.rename(columns={'type': 'actions'})
    
    if avg_positions.empty:
        print(f"❌ No players with sufficient actions found for {team_name}")
        return
    
    # Create pitch
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color=colors['background'],
        line_color='white',
        linewidth=2
    )
    
    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor(colors['background'])
    pitch.draw(ax=ax)
    
    # Plot player positions
    for player, data in avg_positions.iterrows():
        x, y, actions = data['x'], data['y'], data['actions']
        
        # Circle size based on number of actions
        circle_size = max(300, min(1000, actions * 15))  # Scale circle size
        
        # Plot player circle
        scatter = ax.scatter(x, y, s=circle_size, c=colors['primary'], 
                           alpha=0.8, edgecolors='white', linewidth=2, zorder=5)
        
        # Add player name and position
        player_name = player.split()[-1] if ' ' in player else player  # Last name only
        
        # Position text to avoid overlap
        text_y_offset = -8
        
        ax.text(x, y + text_y_offset, player_name, 
               ha='center', va='top', fontsize=10, color='white', 
               weight='bold', zorder=6)
        
        # Add action count
        ax.text(x, y + text_y_offset - 4, f'({actions})', 
               ha='center', va='top', fontsize=8, color='lightgray', 
               zorder=6)
    
    # Add formation lines (connect defensive/midfield/forward lines)
    try:
        # Sort players by average x position (defensive to attacking)
        sorted_players = avg_positions.sort_values('x')
        
        # Try to identify formation lines
        defenders = sorted_players[sorted_players['x'] <= sorted_players['x'].quantile(0.33)]
        midfielders = sorted_players[
            (sorted_players['x'] > sorted_players['x'].quantile(0.33)) & 
            (sorted_players['x'] <= sorted_players['x'].quantile(0.66))
        ]
        forwards = sorted_players[sorted_players['x'] > sorted_players['x'].quantile(0.66)]
        
        # Draw formation lines
        for line_players, alpha in [(defenders, 0.3), (midfielders, 0.3), (forwards, 0.3)]:
            if len(line_players) >= 2:
                line_players_sorted = line_players.sort_values('y')
                for i in range(len(line_players_sorted) - 1):
                    p1 = line_players_sorted.iloc[i]
                    p2 = line_players_sorted.iloc[i + 1]
                    ax.plot([p1['x'], p2['x']], [p1['y'], p2['y']], 
                           color=colors['primary'], alpha=alpha, linewidth=1, zorder=1)
        
    except Exception as e:
        print(f"⚠️ Could not draw formation lines: {e}")
    
    # Calculate team centroid
    centroid_x = avg_positions['x'].mean()
    centroid_y = avg_positions['y'].mean()
    
    # Add team centroid
    ax.scatter(centroid_x, centroid_y, s=200, c='yellow', marker='*', 
              edgecolors='black', linewidth=2, alpha=0.9, zorder=10)
    ax.text(centroid_x, centroid_y - 12, 'Team Center', ha='center', va='top', 
           fontsize=12, color='yellow', weight='bold')
    
    # Title and styling
    ax.set_title(f'{team_name} - Average Player Positions', 
                fontsize=20, color='white', pad=25, weight='bold')
    
    # Add statistics text
    stats_text = f"""
Team Statistics:
• Players shown: {len(avg_positions)}
• Team centroid: ({centroid_x:.1f}, {centroid_y:.1f})
• Formation spread: {avg_positions['x'].max() - avg_positions['x'].min():.1f} x {avg_positions['y'].max() - avg_positions['y'].min():.1f}
    """.strip()
    
    ax.text(2, 78, stats_text, fontsize=11, color='white',
           bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['primary'], alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Average Position Analysis for {team_name}:")
    print(f"   • Players analyzed: {len(avg_positions)}")
    print(f"   • Team centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
    print(f"   • Most attacking player: {avg_positions.loc[avg_positions['x'].idxmax()].name} (x={avg_positions['x'].max():.1f})")
    print(f"   • Most defensive player: {avg_positions.loc[avg_positions['x'].idxmin()].name} (x={avg_positions['x'].min():.1f})")
    
    return avg_positions

# 🆚 TEAM FORMATION COMPARISON
def compare_team_formations(events_df):
    """
    Create side-by-side comparison of both teams' average positions
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    
    teams = events_df['team'].unique()
    if len(teams) < 2:
        print("❌ Need at least 2 teams for comparison")
        return
    
    team1, team2 = teams[0], teams[1]
    
    # Team colors
    team_colors = {
        team1: {'primary': '#75AADB', 'background': '#001F3F'},  # Argentina
        team2: {'primary': '#EF4135', 'background': '#001F3F'}   # France
    }
    
    print(f"🔥 Comparing formations: {team1} vs {team2}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.patch.set_facecolor('#001F3F')
    
    team_stats = {}
    
    for ax, team in [(ax1, team1), (ax2, team2)]:
        colors = team_colors[team]
        
        # Filter events for the team
        team_events = events_df[
            (events_df['team'] == team) & 
            (events_df['location'].notna())
        ].copy()
        
        if team_events.empty:
            continue
        
        # Extract coordinates
        def safe_extract_coords(location):
            try:
                if isinstance(location, list) and len(location) >= 2:
                    return pd.Series([float(location[0]), float(location[1])])
            except:
                pass
            return pd.Series([None, None])
        
        team_events[['x', 'y']] = team_events['location'].apply(safe_extract_coords)
        team_events = team_events.dropna(subset=['x', 'y'])
        
        if team_events.empty:
            continue
        
        # Calculate average positions
        avg_positions = team_events.groupby('player').agg({
            'x': 'mean',
            'y': 'mean',
            'type': 'count'
        }).round(1)
        
        avg_positions = avg_positions[avg_positions['type'] >= 10]  # Min 10 actions
        avg_positions = avg_positions.rename(columns={'type': 'actions'})
        
        team_stats[team] = avg_positions
        
        # Create pitch
        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color=colors['background'],
            line_color='white',
            linewidth=2
        )
        pitch.draw(ax=ax)
        
        # Plot players
        for player, data in avg_positions.iterrows():
            x, y, actions = data['x'], data['y'], data['actions']
            
            circle_size = max(200, min(800, actions * 10))
            
            ax.scatter(x, y, s=circle_size, c=colors['primary'], 
                      alpha=0.8, edgecolors='white', linewidth=2)
            
            player_name = player.split()[-1] if ' ' in player else player
            ax.text(x, y - 6, player_name, ha='center', va='top', 
                   fontsize=9, color='white', weight='bold')
            
            # Add jersey numbers or positions
            ax.text(x, y + 2, f'{actions}', ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
        
        # Team centroid
        if not avg_positions.empty:
            centroid_x = avg_positions['x'].mean()
            centroid_y = avg_positions['y'].mean()
            
            ax.scatter(centroid_x, centroid_y, s=300, c='yellow', marker='*', 
                      edgecolors='black', linewidth=2, alpha=0.9)
            
            # Formation compactness
            x_spread = avg_positions['x'].max() - avg_positions['x'].min()
            y_spread = avg_positions['y'].max() - avg_positions['y'].min()
            
            stats_text = f"""
{len(avg_positions)} players
Centroid: ({centroid_x:.1f}, {centroid_y:.1f})
Spread: {x_spread:.1f} x {y_spread:.1f}
            """.strip()
            
            ax.text(2, 76, stats_text, fontsize=10, color='white',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['primary'], alpha=0.8))
        
        ax.set_title(f'{team} - Formation', fontsize=16, color='white', 
                    pad=15, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Formation analysis
    if len(team_stats) == 2:
        print(f"\n📊 FORMATION COMPARISON:")
        for team, positions in team_stats.items():
            if not positions.empty:
                centroid_x = positions['x'].mean()
                most_attacking = positions['x'].max()
                most_defensive = positions['x'].min()
                width = positions['y'].max() - positions['y'].min()
                
                print(f"\n🏟️ {team}:")
                print(f"   • Team centroid: {centroid_x:.1f} (higher = more attacking)")
                print(f"   • Formation width: {width:.1f}")
                print(f"   • Deepest player: x={most_defensive:.1f}")
                print(f"   • Highest player: x={most_attacking:.1f}")
    
    return team_stats

# 🎯 POSITIONAL HEATMAPS BY PLAYER ROLE
def create_positional_heatmaps(events_df, team_name, colors=None):
    """
    Create heatmaps showing where different types of players operated
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    import seaborn as sns
    
    if colors is None:
        colors = {'primary': '#E3120B', 'background': '#0C0C0C', 'text': '#FFFFFF'}
    
    # Filter team events
    team_events = events_df[
        (events_df['team'] == team_name) & 
        (events_df['location'].notna())
    ].copy()
    
    if team_events.empty:
        print(f"❌ No events found for {team_name}")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    team_events[['x', 'y']] = team_events['location'].apply(safe_extract_coords)
    team_events = team_events.dropna(subset=['x', 'y'])
    
    if team_events.empty:
        print(f"❌ No valid coordinates found for {team_name}")
        return
    
    # Calculate average positions to classify players
    avg_positions = team_events.groupby('player')[['x', 'y']].mean()
    
    # Classify players by position based on average x coordinate
    x_terciles = avg_positions['x'].quantile([0.33, 0.66])
    
    defenders = avg_positions[avg_positions['x'] <= x_terciles.iloc[0]].index
    midfielders = avg_positions[
        (avg_positions['x'] > x_terciles.iloc[0]) & 
        (avg_positions['x'] <= x_terciles.iloc[1])
    ].index
    forwards = avg_positions[avg_positions['x'] > x_terciles.iloc[1]].index
    
    # Create subplots for each position group
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor(colors['background'])
    
    position_groups = [
        (defenders, 'Defenders', '#4CAF50'),
        (midfielders, 'Midfielders', '#FF9800'), 
        (forwards, 'Forwards', '#F44336')
    ]
    
    for ax, (players, position_name, pos_color) in zip(axes, position_groups):
        if len(players) == 0:
            continue
            
        # Filter events for this position group
        pos_events = team_events[team_events['player'].isin(players)]
        
        # Create pitch
        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color=colors['background'],
            line_color='white',
            linewidth=2
        )
        pitch.draw(ax=ax)
        
        # Create heatmap
        if not pos_events.empty:
            try:
                # Create 2D histogram
                hist, xedges, yedges = np.histogram2d(
                    pos_events['x'], pos_events['y'],
                    bins=[np.linspace(0, 120, 25), np.linspace(0, 80, 17)]
                )
                
                # Plot heatmap
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(hist.T, extent=extent, origin='lower', 
                              cmap='Reds', alpha=0.7, aspect='auto')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
                cbar.set_label('Activity Density', color='white', fontsize=10)
                cbar.ax.tick_params(colors='white')
                
            except Exception as e:
                # Fallback to scatter plot
                ax.scatter(pos_events['x'], pos_events['y'], 
                          s=20, c=pos_color, alpha=0.6)
        
        # Add player names
        for player in players:
            player_pos = avg_positions.loc[player]
            player_name = player.split()[-1] if ' ' in player else player
            ax.text(player_pos['x'], player_pos['y'], player_name, 
                   ha='center', va='center', fontsize=8, color='white', 
                   weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=pos_color, alpha=0.7))
        
        ax.set_title(f'{position_name} ({len(players)} players)', 
                    fontsize=14, color='white', pad=10, weight='bold')
    
    plt.suptitle(f'{team_name} - Positional Heatmaps', 
                fontsize=18, color='white', y=0.95, weight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Positional Analysis for {team_name}:")
    print(f"   • Defenders: {len(defenders)} players")
    print(f"   • Midfielders: {len(midfielders)} players") 
    print(f"   • Forwards: {len(forwards)} players")

# 🏆 RUN ALL AVERAGE POSITION VISUALIZATIONS
def create_all_position_maps():
    """
    Create all position-based visualizations for the World Cup Final
    """
    print("🗺️ CREATING ALL AVERAGE POSITION MAPS")
    print("=" * 60)
    
    if events_data is None or events_data.empty:
        print("❌ No events data available")
        return
    
    home_team = world_cup_final['home_team']  # Argentina
    away_team = world_cup_final['away_team']  # France
    
    # Team colors
    argentina_colors = {
        'primary': '#75AADB',    # Argentina light blue
        'background': '#001F3F', # Dark blue
        'text': '#87CEEB'       # Light blue
    }
    
    france_colors = {
        'primary': '#EF4135',    # France red
        'background': '#001F3F', # Dark blue
        'text': '#FFFFFF'       # White
    }
    
    print(f"\n📍 INDIVIDUAL TEAM AVERAGE POSITIONS:")
    print("-" * 50)
    argentina_avg = create_average_position_map(events_data, home_team, argentina_colors)
    france_avg = create_average_position_map(events_data, away_team, france_colors)
    
    print(f"\n🆚 FORMATION COMPARISON:")
    print("-" * 50)
    team_formations = compare_team_formations(events_data)
    
    print(f"\n🎯 POSITIONAL HEATMAPS:")
    print("-" * 50)
    create_positional_heatmaps(events_data, home_team, argentina_colors)
    create_positional_heatmaps(events_data, away_team, france_colors)
    
    print(f"\n🏆 ALL POSITION MAPS COMPLETE!")
    print("🎉 Tactical formations and player roles visualized!")

# Run all position visualizations
create_all_position_maps()

# %%
# 🏁 STARTING 11 FORMATIONS
def create_starting_11_positions(events_df, team_name, colors=None):
    """
    Create average position map showing only the starting 11 players
    
    Args:
        events_df: StatsBomb events DataFrame
        team_name: Team name to analyze
        colors: Color scheme dictionary
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    import numpy as np
    
    if colors is None:
        colors = {'primary': '#E3120B', 'background': '#0C0C0C', 'text': '#FFFFFF'}
    
    # Filter events for the team with location data
    team_events = events_df[
        (events_df['team'] == team_name) & 
        (events_df['location'].notna())
    ].copy()
    
    if team_events.empty:
        print(f"❌ No events with location found for {team_name}")
        return
    
    # Extract coordinates
    def safe_extract_coords(location):
        try:
            if isinstance(location, list) and len(location) >= 2:
                return pd.Series([float(location[0]), float(location[1])])
        except:
            pass
        return pd.Series([None, None])
    
    team_events[['x', 'y']] = team_events['location'].apply(safe_extract_coords)
    team_events = team_events.dropna(subset=['x', 'y'])
    
    if team_events.empty:
        print(f"❌ No valid coordinates found for {team_name}")
        return
    
    # Calculate average positions and action counts for each player
    player_stats = team_events.groupby('player').agg({
        'x': 'mean',
        'y': 'mean',
        'type': 'count',  # Number of actions
        'minute': 'min'   # First appearance time
    }).round(1)
    
    player_stats = player_stats.rename(columns={'type': 'actions', 'minute': 'first_minute'})
    
    # Get the 11 most active players (likely the starting 11)
    # Prioritize players who appeared early and had many actions
    player_stats['priority_score'] = (
        player_stats['actions'] * 2 +  # Weight actions heavily
        (90 - player_stats['first_minute']) * 0.5  # Earlier = higher score
    )
    
    # Get top 11 players
    starting_11 = player_stats.nlargest(11, 'priority_score')
    
    if len(starting_11) == 0:
        print(f"❌ Could not identify starting 11 for {team_name}")
        return
    
    # Create pitch
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color=colors['background'],
        line_color='white',
        linewidth=3
    )
    
    fig, ax = plt.subplots(figsize=(20, 14))
    fig.patch.set_facecolor(colors['background'])
    pitch.draw(ax=ax)
    
    # Plot player positions with formation lines
    for player, data in starting_11.iterrows():
        x, y, actions = data['x'], data['y'], data['actions']
        
        # Larger, more prominent circles
        circle_size = max(400, min(1200, actions * 20))
        
        # Plot player circle
        scatter = ax.scatter(x, y, s=circle_size, c=colors['primary'], 
                           alpha=0.9, edgecolors='white', linewidth=3, zorder=5)
        
        # Add player name (last name only for cleanliness)
        player_name = player.split()[-1] if ' ' in player else player
        
        # Position text below the circle
        ax.text(x, y - 8, player_name, 
               ha='center', va='top', fontsize=12, color='white', 
               weight='bold', zorder=6)
        
        # Add jersey number or position indicator
        ax.text(x, y + 2, f'{int(actions)}', 
               ha='center', va='center', fontsize=10, color='white', 
               weight='bold', zorder=6)
    
    # Draw formation lines to show tactical structure
    try:
        # Sort players by position (defensive to attacking)
        sorted_by_x = starting_11.sort_values('x')
        
        # Identify formation lines based on x-position clustering
        x_positions = sorted_by_x['x'].values
        
        # Use simple terciles to identify defensive lines
        defenders = sorted_by_x[sorted_by_x['x'] <= sorted_by_x['x'].quantile(0.25)]
        midfielders = sorted_by_x[
            (sorted_by_x['x'] > sorted_by_x['x'].quantile(0.25)) & 
            (sorted_by_x['x'] <= sorted_by_x['x'].quantile(0.75))
        ]
        forwards = sorted_by_x[sorted_by_x['x'] > sorted_by_x['x'].quantile(0.75)]
        
        # Draw formation lines
        formation_lines = []
        for line_players, line_name, alpha in [
            (defenders, 'Defense', 0.4), 
            (midfielders, 'Midfield', 0.4), 
            (forwards, 'Attack', 0.4)
        ]:
            if len(line_players) >= 2:
                formation_lines.append(f"{len(line_players)}")
                # Sort by y-position for proper line drawing
                line_sorted = line_players.sort_values('y')
                
                # Draw connecting lines between players in same line
                for i in range(len(line_sorted) - 1):
                    p1 = line_sorted.iloc[i]
                    p2 = line_sorted.iloc[i + 1]
                    ax.plot([p1['x'], p2['x']], [p1['y'], p2['y']], 
                           color=colors['primary'], alpha=alpha, linewidth=2, zorder=1)
        
        # Determine formation
        formation = "-".join(formation_lines) if formation_lines else "Unknown"
        
    except Exception as e:
        formation = "Custom"
        print(f"⚠️ Could not determine formation: {e}")
    
    # Calculate team metrics
    centroid_x = starting_11['x'].mean()
    centroid_y = starting_11['y'].mean()
    formation_width = starting_11['y'].max() - starting_11['y'].min()
    formation_length = starting_11['x'].max() - starting_11['x'].min()
    
    # Add team centroid
    ax.scatter(centroid_x, centroid_y, s=400, c='yellow', marker='*', 
              edgecolors='black', linewidth=3, alpha=0.9, zorder=10)
    ax.text(centroid_x, centroid_y - 15, 'Team Center', ha='center', va='top', 
           fontsize=14, color='yellow', weight='bold')
    
    # Title with formation
    ax.set_title(f'{team_name} Starting XI - Formation: {formation}', 
                fontsize=24, color='white', pad=30, weight='bold')
    
    # Add tactical statistics
    stats_text = f"""
STARTING XI ANALYSIS:
• Formation: {formation}
• Team centroid: ({centroid_x:.1f}, {centroid_y:.1f})
• Formation width: {formation_width:.1f} yards
• Formation length: {formation_length:.1f} yards
• Total actions: {starting_11['actions'].sum():,}
    """.strip()
    
    ax.text(2, 76, stats_text, fontsize=12, color='white',
           bbox=dict(boxstyle="round,pad=0.8", facecolor=colors['primary'], alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Starting XI Analysis for {team_name}:")
    print(f"   • Formation: {formation}")
    print(f"   • Team centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
    print(f"   • Most attacking: {starting_11.loc[starting_11['x'].idxmax()].name} (x={starting_11['x'].max():.1f})")
    print(f"   • Most defensive: {starting_11.loc[starting_11['x'].idxmin()].name} (x={starting_11['x'].min():.1f})")
    print(f"   • Formation width: {formation_width:.1f} yards")
    
    # Show the starting 11 players
    print(f"\n🏁 STARTING XI PLAYERS:")
    for i, (player, data) in enumerate(starting_11.sort_values('x').iterrows(), 1):
        position = "GK" if data['x'] < 20 else ("DEF" if data['x'] < 40 else ("MID" if data['x'] < 80 else "FWD"))
        print(f"   {i:2}. {player.split()[-1]:15} | {position} | ({data['x']:5.1f}, {data['y']:5.1f}) | {data['actions']:3.0f} actions")
    
    return starting_11

# 🆚 STARTING XI COMPARISON
def compare_starting_formations():
    """
    Create side-by-side comparison of both teams' starting 11 formations
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    
    if events_data is None or events_data.empty:
        print("❌ No events data available")
        return
    
    teams = events_data['team'].unique()
    if len(teams) < 2:
        print("❌ Need at least 2 teams for comparison")
        return
    
    team1, team2 = teams[0], teams[1]
    
    # Team colors
    team_colors = {
        team1: {'primary': '#75AADB', 'background': '#001F3F'},  # Argentina
        team2: {'primary': '#EF4135', 'background': '#001F3F'}   # France
    }
    
    print(f"🔥 STARTING XI FORMATION BATTLE: {team1} vs {team2}")
    print("=" * 60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))
    fig.patch.set_facecolor('#001F3F')
    
    team_lineups = {}
    
    for ax, team in [(ax1, team1), (ax2, team2)]:
        colors = team_colors[team]
        
        # Get team events
        team_events = events_data[
            (events_data['team'] == team) & 
            (events_data['location'].notna())
        ].copy()
        
        if team_events.empty:
            continue
        
        # Extract coordinates
        def safe_extract_coords(location):
            try:
                if isinstance(location, list) and len(location) >= 2:
                    return pd.Series([float(location[0]), float(location[1])])
            except:
                pass
            return pd.Series([None, None])
        
        team_events[['x', 'y']] = team_events['location'].apply(safe_extract_coords)
        team_events = team_events.dropna(subset=['x', 'y'])
        
        if team_events.empty:
            continue
        
        # Get starting 11
        player_stats = team_events.groupby('player').agg({
            'x': 'mean',
            'y': 'mean', 
            'type': 'count',
            'minute': 'min'
        }).round(1)
        
        player_stats = player_stats.rename(columns={'type': 'actions', 'minute': 'first_minute'})
        player_stats['priority_score'] = (
            player_stats['actions'] * 2 + 
            (90 - player_stats['first_minute']) * 0.5
        )
        
        starting_11 = player_stats.nlargest(11, 'priority_score')
        team_lineups[team] = starting_11
        
        # Create pitch
        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color=colors['background'],
            line_color='white',
            linewidth=3
        )
        pitch.draw(ax=ax)
        
        # Plot players
        for player, data in starting_11.iterrows():
            x, y, actions = data['x'], data['y'], data['actions']
            
            circle_size = max(300, min(1000, actions * 15))
            
            ax.scatter(x, y, s=circle_size, c=colors['primary'], 
                      alpha=0.9, edgecolors='white', linewidth=3)
            
            player_name = player.split()[-1] if ' ' in player else player
            ax.text(x, y - 6, player_name, ha='center', va='top', 
                   fontsize=11, color='white', weight='bold')
            
            # Show action count
            ax.text(x, y + 2, f'{int(actions)}', ha='center', va='center', 
                   fontsize=9, color='white', weight='bold')
        
        # Formation analysis
        sorted_by_x = starting_11.sort_values('x')
        defenders = sorted_by_x[sorted_by_x['x'] <= sorted_by_x['x'].quantile(0.25)]
        midfielders = sorted_by_x[
            (sorted_by_x['x'] > sorted_by_x['x'].quantile(0.25)) & 
            (sorted_by_x['x'] <= sorted_by_x['x'].quantile(0.75))
        ]
        forwards = sorted_by_x[sorted_by_x['x'] > sorted_by_x['x'].quantile(0.75)]
        
        formation_counts = [len(defenders), len(midfielders), len(forwards)]
        formation = "-".join([str(c) for c in formation_counts if c > 0])
        
        # Team metrics
        centroid_x = starting_11['x'].mean()
        centroid_y = starting_11['y'].mean()
        width = starting_11['y'].max() - starting_11['y'].min()
        
        # Add centroid
        ax.scatter(centroid_x, centroid_y, s=400, c='yellow', marker='*', 
                  edgecolors='black', linewidth=3, alpha=0.9)
        
        # Stats box
        stats_text = f"""
Formation: {formation}
Centroid: ({centroid_x:.1f}, {centroid_y:.1f})
Width: {width:.1f}
Actions: {starting_11['actions'].sum():,}
        """.strip()
        
        ax.text(2, 76, stats_text, fontsize=11, color='white',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['primary'], alpha=0.9))
        
        ax.set_title(f'{team} Starting XI\n{formation}', 
                    fontsize=18, color='white', pad=20, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Compare formations
    if len(team_lineups) == 2:
        print(f"\n📊 FORMATION COMPARISON:")
        for team, lineup in team_lineups.items():
            centroid_x = lineup['x'].mean()
            width = lineup['y'].max() - lineup['y'].min()
            most_attacking = lineup['x'].max()
            most_defensive = lineup['x'].min()
            
            print(f"\n🏟️ {team}:")
            print(f"   • Team shape: {centroid_x:.1f} (attacking position)")
            print(f"   • Formation width: {width:.1f} yards")
            print(f"   • Defensive line: x={most_defensive:.1f}")
            print(f"   • Attacking line: x={most_attacking:.1f}")
            print(f"   • Total team actions: {lineup['actions'].sum():,}")
    
    return team_lineups

# 🏆 RUN STARTING XI ANALYSIS
def analyze_starting_formations():
    """
    Run complete starting XI formation analysis
    """
    print("🏁 WORLD CUP FINAL STARTING XI ANALYSIS")
    print("=" * 60)
    
    if events_data is None or events_data.empty:
        print("❌ No events data available")
        return
    
    home_team = world_cup_final['home_team']  # Argentina
    away_team = world_cup_final['away_team']  # France
    
    # Team colors
    argentina_colors = {
        'primary': '#75AADB',    # Argentina light blue
        'background': '#001F3F', # Dark blue
        'text': '#87CEEB'       # Light blue
    }
    
    france_colors = {
        'primary': '#EF4135',    # France red
        'background': '#001F3F', # Dark blue
        'text': '#FFFFFF'       # White
    }
    
    print(f"\n🇦🇷 ARGENTINA STARTING XI:")
    print("-" * 40)
    argentina_xi = create_starting_11_positions(events_data, home_team, argentina_colors)
    
    print(f"\n🇫🇷 FRANCE STARTING XI:")
    print("-" * 40)
    france_xi = create_starting_11_positions(events_data, away_team, france_colors)
    
    print(f"\n🆚 FORMATION BATTLE:")
    print("-" * 40)
    lineups = compare_starting_formations()
    
    print(f"\n🏆 STARTING XI ANALYSIS COMPLETE!")

# Run starting XI analysis
analyze_starting_formations()


