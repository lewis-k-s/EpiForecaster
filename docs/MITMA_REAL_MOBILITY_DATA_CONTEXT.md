# Origin-Destination (OD) Mobility Data Documentation

## Overview
This documentation describes the structure and content of the origin-destination mobility data files from MITMA (Ministerio de Transportes, Movilidad y Agenda Urbana - Spanish Ministry of Transport, Mobility and Urban Agenda). The dataset contains daily mobility flows measured in person-hours between different geographic zones within Catalonia, Spain.

## File Structure and Naming Convention

### Directory
Data files are located in: `/home/lknoxstr/code/meprecisa/sage_experiments/data/files/daily_dynpop_mitma/`

### Naming Pattern
Files follow the naming convention:
```
mitma_mov_cat.daily_personhours.YYYY-MM-DD_YYYY-MM-DD.nc
```

Where:
- `mitma_mov_cat`: Indicates MITMA mobility data for Catalonia
- `daily_personhours`: Data represents daily mobility measured in person-hours
- Date ranges indicate the temporal coverage of each file (monthly files)

### File Format
All files are in NetCDF format (.nc) suitable for scientific data analysis.

## Data Dimensions and Structure

### Dimensions
Each NetCDF file contains the following dimensions:
- **time**: Variable length depending on the month (16-31 days)
- **home**: 584 origin zones
- **destination**: 585 destination zones (584 same as home + 1 external zone)

### Variables

#### Main Data Variable
- **person_hours(time, home, destination)**
  - Type: double precision floating point
  - Units: Person-hours per day
  - Fill value: NaN for missing data
  - Description: Number of person-hours spent by people from each home zone in each destination zone per day

#### Coordinate Variables
- **home(home)**
  - Type: string
  - Description: Identifiers for origin zones (where people reside)
  
- **destination(destination)**
  - Type: string
  - Description: Identifiers for destination zones (where people travel to)
  
- **time(time)**
  - Type: int64
  - Units: "days since YYYY-MM-DD 00:00:00" (varies by file)
  - Calendar: "proleptic_gregorian"
  - Description: Days since the reference date for each file

## Spatial Units and Zone Codes

### Code Format
Zone identifiers follow Spanish administrative coding systems:

#### Regular Municipality Codes
- **5-digit codes** (e.g., "08001", "43123"): Standard INE municipality codes
  - First 2 digits: Province code (08 = Barcelona, 43 = Tarragona, 25 = Lleida, 17 = Girona)
  - Last 3 digits: Municipality identifier within province

#### Sub-municipal Zones
- **7-digit codes** (e.g., "0800601", "4314805"): Sub-municipal divisions for large cities
  - First 5 digits: Municipality code
  - Last 2 digits: Sub-zone identifier

#### Metropolitan Area Codes
- **Codes with "_AM" suffix** (e.g., "08010_AM", "43142_AM"): Metropolitan area aggregations
- **Codes with "_AD" suffix** (e.g., "4312310_AD"): Administrative district aggregations

#### External Zone
- **"out_cat"**: Represents movements to/from outside Catalonia

### Geographic Coverage
The dataset covers the autonomous community of Catalonia, Spain, with zones corresponding to:
- 584 home zones (places of residence)
- 584 destination zones (same geographic areas as home zones)
- 1 additional "out_cat" destination for external mobility

## Temporal Coverage and Resolution

### Time Period
Available data spans from February 2020 to April 2021, organized in monthly files:
- 2020-02-01 to 2020-02-29 (leap year, 29 days, but file shows 16 time steps)
- 2020-03-01 to 2020-03-31 (31 days)
- 2020-04-01 to 2020-04-30 (30 days)
- ... continuing through 2021-04-30

### Temporal Resolution
- **Daily resolution**: Each time step represents one day
- **Reference time**: Each file uses its own reference date (typically the first day of the month)
- **Time encoding**: Integer days since reference date

### Data Coverage Note
The February 2020 file contains only 16 time steps rather than the expected 29 days, suggesting partial month coverage or data availability limitations.

## Data Units and Interpretation

### Person-Hours
The fundamental unit of measurement is "person-hours per day":
- Represents the total time (in hours) that people from origin zone spend in destination zone per day
- Calculated as: Number of people × Hours spent in destination
- Example: If 100 people from zone A spend 2 hours each in zone B on a given day, the value would be 200 person-hours

### Data Values
- **Typical range**: 0 to several hundred thousand person-hours
- **Zero values**: Common, indicating no recorded movement between zone pairs
- **Large values**: Diagonal elements (home = destination) typically show highest values, representing time spent in home zones
- **Missing data**: Represented as NaN (Not a Number)

## Data Context and Applications

### Source
MITMA (Ministerio de Transportes, Movilidad y Agenda Urbana) - Spanish Ministry of Transport, Mobility and Urban Agenda

### Methodology
Data likely derived from mobile phone location analytics, providing aggregate mobility patterns while preserving privacy through anonymization and aggregation.

### Potential Uses
- Urban planning and transportation analysis
- COVID-19 mobility impact studies (given 2020-2021 timeframe)
- Regional economic flow analysis
- Population dynamics research
- Infrastructure planning and optimization

### Data Quality Considerations
- Temporal gaps may exist (e.g., February 2020 partial coverage)
- Privacy protection may result in small values being suppressed or aggregated
- Mobile phone penetration rates may affect representativeness
- Administrative boundary changes could affect zone consistency

## Technical Specifications

### File Sizes
Files range from approximately 40-85 MB each, with larger files for months with more days.

### Coordinate Reference System
Geographic zones reference Spanish administrative divisions (INE codes), implicitly using the Spanish coordinate reference systems. Related geographic files use EPSG:3042 (suitable for Catalonia region).

### Data Access
Files can be accessed using standard NetCDF libraries in Python (xarray, netCDF4), R (ncdf4), or command-line tools (ncdump, ncks).

---

*Documentation generated based on analysis of MITMA daily mobility data files. For the most current information about data methodology and updates, consult official MITMA sources.*