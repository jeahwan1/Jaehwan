from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LineupPlayer:
    order: int
    name: str
    team: str


@dataclass
class TeamLineup:
    team: str
    starter_pitcher: str
    batters: list[LineupPlayer] = field(default_factory=list)


@dataclass
class GameLineup:
    game_id: str
    game_time: str
    away: TeamLineup
    home: TeamLineup


@dataclass
class MatchupStat:
    pitcher: str
    batter: str
    pa: int
    ab: int
    hits: int
    hr: int
    bb: int
    so: int

    @property
    def avg(self) -> float:
        return round(self.hits / self.ab, 3) if self.ab else 0.0


@dataclass
class GameAnalysis:
    game_id: str
    title: str
    notes: list[str]
