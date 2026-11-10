# UE Mission Profiles for Edge Image Scheduling (Draft)

Status: draft from web search and 3GPP/ITU index pages. Clause numbers marked "unverified" were not confirmed in the document text. No runtime or experiment claims are made here.

Traffic class: **U** = mostly urgent/disaster-relevant, **E** = mostly everyday, **M** = mixed.

## 1. Profile table

| id | device family | sub-type | typical image content | mission paragraph (instruction to UE) | class | source |
|---|---|---|---|---|---|---|
| 1 | UAV | disaster-recovery drone | aerial RGB/thermal stills of collapsed structures, flood extent, people | Capture and upload the smallest set of frames that shows survivors, blocked routes or structural damage in the assigned grid. Prioritise frames with new hazards and send them before routine coverage frames. | U | TS 22.125 cl. 6 (UAV usages), TR 22.829 (live video, version unverified) |
| 2 | UAV | public-safety wide-area surveillance drone | 4K video of crowds, vehicles, perimeters | Stream the assigned area at the agreed quality and flag frames where a tracked object enters a restricted zone. Keep the uplink within the rate the network grants. | M | TS 22.125 cl. 7 performance table (4x4K AI surveillance row, unverified exact table ID) |
| 3 | UAV | cinematography / live-broadcast drone | 4K/8K live video of events and sport | Deliver a continuous live stream at the requested resolution and frame rate with the lowest achievable end-to-end delay. Report any drop in throughput so the scheduler can adapt bitrate. | E | TS 22.125 cl. 7 (8K live broadcast row, 100 Mbps UAV uplink, per search excerpt); TS 22.263 (VIAPA) |
| 4 | UAV | agriculture scan / spray drone | multispectral and RGB crop plots, weed or disease patches | Send geotagged image tiles of each survey lane, with a low-quality preview for every tile and full resolution only for tiles flagged as anomalous. | E | TR 22.843 cl. 4-5 (UAV Phase 3 use cases, unverified clause) |
| 5 | UAV | delivery drone (BVLOS) | landing-zone frames, obstacle and route imagery | Upload landing-site and obstacle frames at a fixed cadence while on route, and send immediately any frame that triggers detect-and-avoid. Report the frame timestamp and position with each image. | M | TR 22.843 (route management, UTM, detect-and-avoid; V19.2.0 text per search) |
| 6 | UAV | infrastructure-inspection drone | close-up images of power lines, bridges, wind-turbine blades | Capture defect-oriented close-up images along the planned path and send the full set after landing or on a scheduled window. Mark frames with suspected cracks or corrosion for priority upload. | E | TR 22.843 (UAV Phase 3); TS 22.125 cl. 6 (unverified clause) |
| 7 | UAV | aerial base station / relay drone (UxN) | relayed images from ground users, no own camera | Relay the images of attached ground devices without re-ordering them, and report the relay load and link state so the scheduler can allocate airtime. | M | TS 22.125 cl. 6.4 (UxNB control, R-6.4-002) |
| 8 | UAV | firefighting thermal drone | thermal IR frames of fire fronts and hotspots | Send thermal frames of the fire perimeter with hotspot annotation first, and provide regular low-rate updates for the rest of the flight. | U | TS 22.125 cl. 6 (unverified clause); TS 22.280 (MCCoRe, public-safety context) |
| 9 | vehicle / ADAS | connected-car cooperative perception | raw or processed camera and sensor frames | Share locally gathered sensor frames with nearby vehicles and the V2X application server, and mark frames that describe an obstacle or hazard for lowest latency. | M | TS 22.186 (Extended Sensors: sensor and live-video sharing; V15.3.0 ETSI copy; current version unverified) |
| 10 | vehicle / VRU | pedestrian or cyclist smartphone safety video | short clips of traffic approach and crossing | Upload short clips only when a collision risk is detected by the device, and keep the rest of the time silent. | U | TS 22.186 (VRU and V2X scenarios; clause unverified) |
| 11 | vehicle / public safety | emergency-vehicle onboard video | dashcam and cabin video during response | Stream the forward and cabin video at the mission-critical priority level, and degrade resolution first when the link weakens. | U | TS 22.280 (MCCoRe); TS 22.281 (MCVideo, version unverified) |
| 12 | vehicle | teleoperated vehicle or platoon-lead video uplink | forward-facing video for remote driving | Maintain a low-latency forward video stream to the remote operator at the rate required for the current speed, and report any latency breach immediately. | M | TS 22.186 (automated-driving and platooning scenarios); TS 22.261 cl. 6.20 (eV2X aspects) |
| 13 | public safety | body-worn camera (handheld) | first-person video of incidents and bystanders | Upload the live stream when the officer or a dispatcher requests it, and buffer the rest of the footage locally until a link is available. | U | TS 22.179 (MCPTT); MCVideo in TS 22.281 (version unverified) |
| 14 | public safety | firefighter / search-and-rescue handheld with thermal camera | thermal frames of smoke-filled rooms, casualties | Send thermal frames of any detected human heat signature with location tags and send the rest on request. | U | TS 22.280 (MCCoRe); TS 22.179 (MCPTT) |
| 15 | fixed CCTV / IoT | urban traffic camera | periodic traffic frames, incident bursts | Send one frame per interval as a background feed and upload a burst of frames when a vehicle stops or a crash-like pattern is detected. | M | TS 22.261 cl. 6.33 (VIAPA scope, general mapping; clause verified as title only) |
| 16 | fixed CCTV / IoT | remote wildlife / environmental camera trap | low-rate still images triggered by motion | Upload a still image only when motion is detected and batch the rest in a daily window to save energy. | E | ITU-R M.2160 Sec. 3 (massive communication usage scenario) |
| 17 | fixed CCTV / IoT | remote wildfire or flood early-warning camera | solar-powered stills of smoke or water rise | Send an alert frame immediately when smoke or water is detected, and otherwise send a low-rate status frame at the agreed interval. | U | ITU-R M.2160 Sec. 3 (ubiquitous connectivity; unverified clause) |
| 18 | industrial | factory machine-vision QC camera | fixed-view images of parts and assemblies | Send each inspected image with its pass or fail result and a timestamp, and send full-resolution images only for failed parts. | E | TS 22.104 (factories of the future, cyber-physical control); 5G-ACIA white paper (non-3GPP) |
| 19 | industrial | warehouse AGV camera | navigation and pallet images | Send navigation frames at the rate needed for localisation, and drop any frame that is superseded by a newer one. | E | TS 22.104 (cyber-physical control, clause unverified) |
| 20 | industrial | industrial cobot vision | bin-picking and part-pose images | Send images for grasp planning with the lowest achievable latency, and send the rest of the sequence after the task completes. | E | TS 22.104; Hexa-X D1.2 (cooperating-robots family, unverified clause) |
| 21 | industrial / hazardous site | teleoperated inspection robot | video of gas, pipe or tunnel sites | Stream live video when the operator is connected, and buffer all other frames locally with anomaly tags for later upload. | M | TS 22.104 (scope only; hazardous-site requirements unverified) |
| 22 | consumer | smartphone photo or video upload | consumer photos, short videos, AI-enhancement inputs | Upload each image as a background task in the order it was captured, with lower priority for images that are not requested by the user. | E | TS 22.261 (consumer multimedia context; clause unverified) |
| 23 | wearable / XR | AR or smart-glasses egocentric video | first-person scene frames for on-edge inference | Send scene frames at the rate the inference task needs, with lower resolution when the link is busy, and skip frames with no change. | E | Hexa-X D1.2 (telepresence and XR family, unverified clause) |
| 24 | consumer / public safety | citizen emergency SOS smartphone | photos of an incident or injury, location-tagged | Send the photo and location at the highest priority as soon as the user confirms the report, and send follow-up frames only when requested. | U | TS 22.179 and TS 22.280 (public-safety context, clause unverified) |
| 25 | NTN / satellite | maritime or remote-area satellite terminal | SAR or camera stills of vessels and people at sea | Send alert imagery at the first available satellite pass and compress the remaining frames to the lowest quality acceptable for triage. | M | TS 22.261 NTN clauses unverified; ITU-R M.2160 Sec. 3 (ISAC and ubiquitous connectivity, unverified clause) |
| 26 | wearable / medical | surgical or telesurgery headset camera | UHD video of the operating field | Stream the operating-field video with the lowest achievable latency and flag any loss of frames so the surgeon can be alerted. | U | TS 22.263 cl. 6 (medical imaging table, telesurgery latency under 20 ms, per search excerpt) |

## 2. Source documents consulted

- TS 22.125 (UAS support, stage 1), section 6 and 7 on itecspec mirror: https://itecspec.com/spec/3gpp-22-125-6-requirements-for-uav-usages and https://itecspec.com/spec/3gpp-22-125-7-performance-requirements (release differs from the ETSI copy; not confirmed against the target release)
- TS 22.125 V17.7.0 (ETSI): https://www.etsi.org/deliver/etsi_ts/122100_122199/122125/17.07.00_60/ts_122125v170700p.pdf
- TS 22.261 clause 6.33 (VIAPA, title only) and 6.20 (eV2X): https://itecspec.com/spec/3gpp-22-261-6-33-video-imaging-and-audio-for-professional-applications and https://itecspec.com/spec/3gpp-22-261-6-20-ev2x-aspects
- TS 22.263 V19.0.0 (VIAPA, ETSI): https://www.etsi.org/deliver/etsi_ts/122200_122299/122263/19.00.00_60/ts_122263v190000p.pdf
- TS 22.186 clause 5 (requirements, itecspec): https://itecspec.com/spec/3gpp-22-186-5-requirements ; V15.3.0 ETSI: https://www.etsi.org/deliver/etsi_ts/122100_122199/122186/15.03.00_60/ts_122186v150300p.pdf
- TS 22.104 V17.7.0 (ETSI): https://www.etsi.org/deliver/etsi_TS/122100_122199/122104/17.07.00_60/ts_122104v170700p.pdf
- TS 22.179 (MCPTT) and TS 22.280 (MCCoRe) database entries: https://www.3gpp.org/DynaReport/22179.htm and https://3gpp.org/dynareport/22280.htm
- TR 22.843 (UAV Phase 3) database entry: https://3gpp.org/dynareport/22843.htm
- TR 22.829 (UAV enhancements, Rel-16) database entry: https://www.3gpp.org/DynaReport/22829.htm (version unverified)
- TR 22.870 (6G use cases) database entry: https://3gpp.org/dynareport/22870.htm ; SA1 6G use-case page: https://www.3gpp.org/technologies/sa1-use-cases-6g
- ITU-R M.2160 (IMT-2030 framework, 11/2023): https://www.itu.int/dms_pubrec/itu-r/rec/m/R-REC-M.2160-0-202311-I!!PDF-E.pdf
- Hexa-X D1.2 (expanded vision and use cases): https://hexa-x.eu/d1-2-expanded-6g-vision-use-cases-and-societal-values-including-aspects-of-sustainability-security-and-spectrum/

## 3. Unverified or gaps

- TR 22.829: sources disagree on version and on whether it covers remote ID. Remote ID is in TS 22.125 cl. 5 and TR 22.825.
- TR 22.870: final publication number and version not confirmed. Only draft V0.3.1 was seen; an SA1 slide reports approval at TSG#111 (Mar 2026).
- TS 22.281 (MCVideo) and TS 22.282 (MCData): titles seen, current versions and clauses not verified.
- NTN service requirements (profile 25) and drone-specific UAV requirements in TS 22.261 clause 6.32: not verified in text.
- Hexa-X D1.2 was checked only through a summary page; no drone or vehicle passage was located.
- Clause numbers from itecspec mirrors may belong to a later release than the one a paper cites.
