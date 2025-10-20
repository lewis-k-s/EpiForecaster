import pandas as pd

# Spanish COVID-19 Lockdown and Restriction Timeline
# Based on official government announcements and Wikipedia timeline

# MAJOR NATIONAL LOCKDOWN (Full mobility restrictions)
NATIONAL_LOCKDOWN = (
    pd.Timestamp("2020-03-15"),  # State of emergency declared
    pd.Timestamp("2020-06-21"),  # State of emergency officially ended
)

# LEGACY LOCKDOWNS (for backwards compatibility)
# Note: The second period was a state of emergency with curfews, not a full lockdown
LOCKDOWNS = [
    (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-21")),  # Corrected end date
    (
        pd.Timestamp("2020-10-25"),
        pd.Timestamp("2021-05-09"),
    ),  # State of emergency with curfews
]

# LOCKDOWN EXIT PHASES (National timeline for gradual reopening)
LOCKDOWN_PHASES = {
    "phase_0": pd.Timestamp("2020-05-04"),  # Preparatory phase
    "phase_1": pd.Timestamp("2020-05-11"),  # First easing (some regions May 25)
    "phase_2": pd.Timestamp("2020-05-25"),  # Second easing (some regions June 1)
    "phase_3": pd.Timestamp("2020-06-08"),  # Third easing
    "new_normal": pd.Timestamp("2020-06-21"),  # Full reopening
}

# REGIONAL LOCKDOWN VARIATIONS
# Key regions had different timelines due to infection rates
REGIONAL_LOCKDOWNS = {
    "madrid": {
        "early_restrictions": (
            pd.Timestamp("2020-03-09"),
            pd.Timestamp("2020-03-15"),
        ),  # School closures
        "national_lockdown": (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-21")),
        "delayed_phase_1": pd.Timestamp("2020-05-25"),  # Madrid delayed to Phase 1
        "october_partial": (
            pd.Timestamp("2020-10-01"),
            pd.Timestamp("2020-10-25"),
        ),  # Partial lockdown
        "state_emergency_2": (pd.Timestamp("2020-10-25"), pd.Timestamp("2021-05-09")),
    },
    "catalonia": {
        "municipal_quarantine": (
            pd.Timestamp("2020-03-12"),
            pd.Timestamp("2020-03-15"),
        ),  # 4 municipalities
        "early_restrictions": (
            pd.Timestamp("2020-03-12"),
            pd.Timestamp("2020-03-15"),
        ),  # School closures
        "national_lockdown": (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-21")),
        "barcelona_delayed": pd.Timestamp("2020-05-25"),  # Barcelona delayed phases
        "state_emergency_2": (pd.Timestamp("2020-10-25"), pd.Timestamp("2021-05-09")),
    },
    "basque_country": {
        "early_restrictions": (
            pd.Timestamp("2020-03-09"),
            pd.Timestamp("2020-03-15"),
        ),  # School closures
        "national_lockdown": (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-21")),
        "state_emergency_2": (pd.Timestamp("2020-10-25"), pd.Timestamp("2021-05-09")),
    },
    "la_rioja": {
        "early_restrictions": (
            pd.Timestamp("2020-03-09"),
            pd.Timestamp("2020-03-15"),
        ),  # School closures
        "national_lockdown": (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-21")),
        "state_emergency_2": (pd.Timestamp("2020-10-25"), pd.Timestamp("2021-05-09")),
    },
}

# STATE OF EMERGENCY PERIODS (Distinguishing full lockdowns from curfew periods)
STATE_OF_EMERGENCY = {
    "first": {
        "period": (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-21")),
        "type": "full_lockdown",
        "description": "Stay-at-home orders, essential activities only",
    },
    "second": {
        "period": (pd.Timestamp("2020-10-25"), pd.Timestamp("2021-05-09")),
        "type": "curfew_restrictions",
        "description": "National curfew (23:00-06:00), regional movement restrictions",
    },
}

# MOBILITY RESTRICTION SEVERITY LEVELS
# For more nuanced analysis of restriction impact
RESTRICTION_LEVELS = {
    "pre_restrictions": {
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2020-03-09"),
        "severity": 0,
        "description": "Normal mobility",
    },
    "early_regional": {
        "start": pd.Timestamp("2020-03-09"),
        "end": pd.Timestamp("2020-03-15"),
        "severity": 2,
        "description": "School closures in some regions",
    },
    "full_lockdown": {
        "start": pd.Timestamp("2020-03-15"),
        "end": pd.Timestamp("2020-06-21"),
        "severity": 5,
        "description": "Stay-at-home orders, essential only",
    },
    "gradual_reopening": {
        "start": pd.Timestamp("2020-05-04"),
        "end": pd.Timestamp("2020-06-21"),
        "severity": 3,  # Average during phases
        "description": "Phased reopening with restrictions",
    },
    "new_normal": {
        "start": pd.Timestamp("2020-06-21"),
        "end": pd.Timestamp("2020-10-25"),
        "severity": 1,
        "description": "Social distancing, some capacity limits",
    },
    "second_wave_restrictions": {
        "start": pd.Timestamp("2020-10-25"),
        "end": pd.Timestamp("2021-05-09"),
        "severity": 3,
        "description": "Curfews and regional restrictions",
    },
    "post_emergency": {
        "start": pd.Timestamp("2021-05-09"),
        "end": pd.Timestamp("2021-12-31"),
        "severity": 1,
        "description": "Regional discretion, vaccination rollout",
    },
}
