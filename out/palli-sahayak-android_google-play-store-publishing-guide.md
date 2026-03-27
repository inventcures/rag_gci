# Palli Sahayak Android App: Google Play Store Publishing Guide

**Date**: 27 March 2026
**App**: Palli Sahayak — Voice AI Clinical Decision Support for Palliative Care
**Package**: `com.pallisahayak.app`
**Repo**: https://github.com/inventcures/palli-sahayak-android

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Generate Signed Release APK/AAB](#2-generate-signed-release-apkaab)
3. [Create Google Play Developer Account](#3-create-google-play-developer-account)
4. [Create the App Listing on Play Console](#4-create-the-app-listing-on-play-console)
5. [Prepare Store Listing Assets](#5-prepare-store-listing-assets)
6. [Content Rating Questionnaire](#6-content-rating-questionnaire)
7. [Pricing and Distribution](#7-pricing-and-distribution)
8. [Data Safety Form](#8-data-safety-form)
9. [Upload the App Bundle](#9-upload-the-app-bundle)
10. [Testing Tracks (Internal/Closed/Open)](#10-testing-tracks)
11. [Production Release](#11-production-release)
12. [Post-Launch Checklist](#12-post-launch-checklist)
13. [EVAH Evaluation-Specific Notes](#13-evah-evaluation-specific-notes)

---

## 1. Prerequisites

### 1.1 Developer Machine Setup

Ensure you have:

```bash
# Java 17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home

# Android SDK
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools

# Verify
java -version        # openjdk 17.x
./gradlew --version  # Gradle 8.7
```

### 1.2 Accounts Required

| Account | URL | Cost | Notes |
|---------|-----|------|-------|
| Google Play Developer | https://play.google.com/console | $25 one-time | Register with the KCDH-A organization account |
| Google Cloud Console | https://console.cloud.google.com | Free | For Firebase Crashlytics, if used |
| Keystore | Generated locally | Free | Must be backed up securely; loss = can never update the app |

### 1.3 Code Readiness Checklist

- [ ] All lint warnings resolved: `./gradlew lint`
- [ ] All unit tests pass: `./gradlew test`
- [ ] Release build compiles: `./gradlew assembleRelease`
- [ ] ProGuard/R8 rules tested (no runtime crashes after minification)
- [ ] `versionCode` and `versionName` set in `app/build.gradle.kts`
- [ ] `BASE_URL` in release build type points to production server
- [ ] Firebase Crashlytics configured (if applicable)
- [ ] No debug logging or hardcoded test credentials in release build

---

## 2. Generate Signed Release APK/AAB

### 2.1 Create a Signing Keystore (First Time Only)

```bash
keytool -genkey -v \
  -keystore palli-sahayak-release.jks \
  -keyalg RSA \
  -keysize 2048 \
  -validity 10000 \
  -alias palli-sahayak

# You will be prompted for:
# - Keystore password (SAVE THIS — cannot be recovered)
# - Key password (can be same as keystore password)
# - Name, Organization, Country (use KCDH-A, Ashoka University, IN)
```

**CRITICAL**: Back up `palli-sahayak-release.jks` to at least two secure locations (encrypted USB, institutional secure storage). If lost, you can never update the app on Play Store.

### 2.2 Configure Signing in Gradle

Create `keystore.properties` in the project root (DO NOT commit this):

```properties
storeFile=../palli-sahayak-release.jks
storePassword=YOUR_KEYSTORE_PASSWORD
keyAlias=palli-sahayak
keyPassword=YOUR_KEY_PASSWORD
```

Add to `app/build.gradle.kts`:

```kotlin
// Load keystore properties
val keystoreProperties = Properties()
val keystoreFile = rootProject.file("keystore.properties")
if (keystoreFile.exists()) {
    keystoreProperties.load(FileInputStream(keystoreFile))
}

android {
    signingConfigs {
        create("release") {
            storeFile = file(keystoreProperties["storeFile"] as String)
            storePassword = keystoreProperties["storePassword"] as String
            keyAlias = keystoreProperties["keyAlias"] as String
            keyPassword = keystoreProperties["keyPassword"] as String
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            // ... existing release config
        }
    }
}
```

### 2.3 Build the Android App Bundle (AAB)

Google Play requires AAB format (not APK) for new apps:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools

# Build release AAB for all sites
./gradlew bundleAllSitesRelease

# Output location:
# app/build/outputs/bundle/allSitesRelease/app-allSites-release.aab
```

### 2.4 Verify the AAB

```bash
# Check AAB size (target: under 150MB, ours should be ~15-20MB)
ls -lh app/build/outputs/bundle/allSitesRelease/app-allSites-release.aab

# Verify signing
jarsigner -verify -verbose -certs \
  app/build/outputs/bundle/allSitesRelease/app-allSites-release.aab

# Optional: Generate APKs from AAB to test locally
java -jar bundletool.jar build-apks \
  --bundle=app/build/outputs/bundle/allSitesRelease/app-allSites-release.aab \
  --output=palli-sahayak.apks \
  --ks=palli-sahayak-release.jks \
  --ks-key-alias=palli-sahayak
```

### 2.5 Also Generate APK for Direct Distribution

For sites where Play Store access is limited (CCHRC Silchar):

```bash
./gradlew assembleAllSitesRelease

# Output:
# app/build/outputs/apk/allSites/release/app-allSites-release.apk
```

---

## 3. Create Google Play Developer Account

### 3.1 Register

1. Go to https://play.google.com/console
2. Sign in with the KCDH-A institutional Google account
3. Pay the $25 one-time registration fee
4. Complete the developer profile:
   - **Developer name**: Koita Centre for Digital Health, Ashoka University
   - **Email**: kcdh@ashoka.edu.in (or appropriate institutional email)
   - **Phone**: Institutional phone number
   - **Website**: https://github.com/inventcures/palli-sahayak-android
5. Verify identity (may require government ID for organizational accounts)

### 3.2 Set Up the Organization (if applicable)

For an organizational account:
1. Navigate to **Settings > Developer account > Organization details**
2. Enter:
   - Organization name: Koita Centre for Digital Health (KCDH-A)
   - Organization address: Ashoka University, Sonipat, Haryana, India
   - Organization website: Ashoka University website
   - DUNS number (if available): Contact Ashoka admin

---

## 4. Create the App Listing on Play Console

### 4.1 Create New App

1. Go to **All apps > Create app**
2. Fill in:
   - **App name**: Palli Sahayak
   - **Default language**: English (India) – en-IN
   - **App or game**: App
   - **Free or paid**: Free
3. Check the declarations:
   - [x] Developer Program Policies
   - [x] US export laws
4. Click **Create app**

### 4.2 Set Up App Details

Navigate to **Grow > Store presence > Main store listing**:

**Short description** (80 chars max):
```
Voice AI for palliative care — clinical guidance in 22 Indian languages
```

**Full description** (4000 chars max):
```
Palli Sahayak ("Companion in Care") is a voice-first AI clinical decision support
tool that provides evidence-based palliative care guidance to community health
workers, family caregivers, and patients in India.

KEY FEATURES:

Voice-First Design
- Ask clinical questions by voice in your native language
- Hands-free operation during home visits
- Supports 22 Indian languages via Sarvam AI

Evidence-Based Guidance
- Answers grounded in WHO Cancer Pain Relief guidelines, Pallium India Clinical
  Handbook, and Max Healthcare Palliative Care Protocols
- Evidence badges (A-E) show response confidence level
- Never provides medication dosages — always recommends physician consultation

Clinical Safety
- Emergency detection in 8 languages with 108 ambulance call button
- Seven conditions trigger automatic handoff to supervising physician
- Monthly clinician review of AI-generated guidance

Offline-First
- Works without internet using cached responses for common queries
- Automatic background sync when connected
- Designed for rural and remote areas with intermittent connectivity

Patient Management
- Track symptoms, medications, and vital signs over time
- Medication reminder voice calls with confirmation
- Care team coordination
- FHIR R4 export for health system interoperability

For ASHA Workers & Caregivers
- Designed for low-literacy users with large touch targets and audio guidance
- PIN-based authentication (no complex passwords)
- Works on budget Android phones (2GB RAM, Android 8.0+)

Built under a Gates Foundation Grand Challenges India grant. Open source (MIT
license): https://github.com/inventcures/palli-sahayak-android

IMPORTANT: This app provides clinical decision support information, not medical
advice. Always consult a qualified healthcare professional for medical decisions.
```

---

## 5. Prepare Store Listing Assets

### 5.1 Required Graphics

| Asset | Size | Format | Description |
|-------|------|--------|-------------|
| App icon | 512 x 512 px | PNG (32-bit, no alpha) | Green gradient with white medical cross + microphone |
| Feature graphic | 1024 x 500 px | PNG or JPG | "Palli Sahayak" text over green gradient with voice waveform |
| Phone screenshots | Min 2, max 8 | 16:9 or 9:16, min 320px, max 3840px | Voice query, dashboard, patient timeline, evidence badge |
| 7-inch tablet screenshots | Optional | Same as phone | If tablet layout differs |
| 10-inch tablet screenshots | Optional | Same as phone | If tablet layout differs |

### 5.2 Screenshots to Capture (minimum 4)

1. **Language Selection Screen** — Shows 9 Indian languages in native scripts
2. **Voice Query Screen** — 96dp microphone button, waveform animation
3. **Query Result with Evidence Badge** — Clinical response with green "Evidence: B" badge
4. **ASHA Dashboard** — Patient list, pending reminders, voice query FAB
5. **Patient Timeline** — Observation history with color-coded categories
6. **Insights Tab** — Consolidated patient insights from Always-On Memory
7. **Emergency Detection** — Red emergency banner with 108 call button
8. **SUS Questionnaire** — Evaluation data collection screen

### 5.3 Generate Screenshots

```bash
# Option 1: Use Android emulator
# Start emulator with a representative device
emulator -avd Pixel_4a -port 5554

# Capture screenshots
adb -s emulator-5554 shell screencap /sdcard/screenshot.png
adb -s emulator-5554 pull /sdcard/screenshot.png ./screenshots/

# Option 2: Use Android Studio Layout Inspector + screenshot tool

# Option 3: Use Maestro for automated screenshot capture
maestro test screenshots.yaml
```

### 5.4 Promotional Video (Optional but Recommended)

- Duration: 30 seconds to 2 minutes
- Show: ASHA worker using voice query during simulated home visit
- Highlight: Multilingual voice, evidence badges, emergency detection
- Upload to YouTube, link from Play Store listing

---

## 6. Content Rating Questionnaire

Navigate to **Policy > App content > Content rating**:

### 6.1 IARC Rating Questionnaire Answers

| Question | Answer | Reason |
|----------|--------|--------|
| Does the app contain violence? | No | Clinical information only |
| Does the app contain sexual content? | No | |
| Does the app contain profanity? | No | |
| Does the app allow user-to-user communication? | No | No chat or messaging between users |
| Does the app share user location? | No | GPS is optional metadata only |
| Does the app allow purchases? | No | Free app |
| Does the app contain ads? | No | |
| Does the app provide health information? | **Yes** | Clinical decision support for palliative care |
| Is the health information for informational purposes only? | **Yes** | Not a substitute for medical advice |

Expected rating: **Rated for everyone** (PEGI 3 / Everyone)

### 6.2 Health App Declaration

When prompted about health claims:
- State that the app provides **clinical decision support information**
- State it does **not provide diagnosis or treatment recommendations**
- State it always recommends **consulting a qualified healthcare professional**
- Include disclaimers in the app and store listing

---

## 7. Pricing and Distribution

Navigate to **Monetize > Pricing**:

1. **Price**: Free
2. **Countries**: India (primary), all countries (secondary)
3. **Contains ads**: No
4. **Content guidelines**: Educational / Medical information

---

## 8. Data Safety Form

Navigate to **Policy > App content > Data safety**:

This is critical for compliance with Google's policies and India's DPDP Act 2023.

### 8.1 Data Collection Declaration

| Data Type | Collected | Shared | Purpose | Optional |
|-----------|-----------|--------|---------|----------|
| Name | Yes | No | App functionality (user profile) | No |
| Phone number | Yes (hashed) | No | User identification | No |
| Health info (symptoms, medications) | Yes | No | Clinical decision support | No |
| App interactions | Yes | No | Evaluation research (EVAH study) | No |
| Device identifiers | Yes | No | Crash reporting | Yes |
| Voice/audio recordings | No* | No | *Temporary only, deleted after processing | N/A |

*Voice audio is processed in real-time and not stored persistently (per EVAH proposal section 10.1).

### 8.2 Security Practices

- [x] Data encrypted in transit (HTTPS)
- [x] Data encrypted at rest (SQLCipher)
- [x] Users can request data deletion (Settings > Delete My Data)
- [x] Committed to follow Play Families Policy (if applicable): No — not designed for children

### 8.3 Data Handling Details

For each data type, specify:

**Health information**:
- **Purpose**: App functionality (clinical decision support)
- **Encrypted**: Yes (SQLCipher at rest, HTTPS in transit)
- **Deletable by user**: Yes (via Settings > Delete My Data)
- **Required for app to function**: Yes
- **Processing location**: India (Sarvam AI for voice processing, backend server)

**Contact info (name)**:
- **Purpose**: App functionality (personalized greeting, care team coordination)
- **Encrypted**: Yes
- **Deletable by user**: Yes

---

## 9. Upload the App Bundle

### 9.1 Navigate to Release

1. Go to **Release > Production** (or Testing track first — recommended)
2. Click **Create new release**

### 9.2 App Signing

Google Play manages app signing for you:

1. **First upload**: Google generates a new app signing key
2. **Recommended**: Use **Google Play App Signing** (let Google manage the signing key)
3. Upload your AAB — Google will re-sign it with the play signing key

If you prefer to manage signing yourself:
1. Export your signing key
2. Upload the signed AAB

### 9.3 Upload the AAB

1. Drag and drop `app-allSites-release.aab` into the upload area
2. Wait for processing (typically 1-5 minutes)
3. Google validates the bundle and reports any issues

### 9.4 Release Notes

```
Version 0.1.0 — Initial Release

Palli Sahayak: Voice-first AI clinical decision support for palliative care

Features:
- Voice clinical queries in 22 Indian languages (Sarvam AI)
- Evidence-based guidance with confidence badges (A-E)
- Emergency detection with 108 ambulance call
- Offline-first with cached common queries
- Patient observation tracking and timeline
- Medication reminder management
- Care team coordination
- FHIR R4 health data export (ABDM compatible)
- EVAH evaluation instrumentation (SUS, clinical vignettes)

Built under Gates Foundation Grand Challenges India grant.
Open source: https://github.com/inventcures/palli-sahayak-android
```

---

## 10. Testing Tracks

### 10.1 Recommended Rollout Strategy

| Track | Audience | Duration | Purpose |
|-------|----------|----------|---------|
| **Internal testing** | 5-10 team members | 1-2 weeks | Catch crashes, verify all flows |
| **Closed testing** | 20-50 site coordinators + PIs | 2-4 weeks | Usability feedback, site-specific issues |
| **Open testing** (optional) | Anyone with link | 1-2 weeks | Broader compatibility testing |
| **Production** | All users | Ongoing | Full public release |

### 10.2 Set Up Internal Testing Track

1. Go to **Release > Testing > Internal testing**
2. Click **Create new release**
3. Upload the AAB
4. Add testers by email (up to 100):
   - Ashish Makani (developer)
   - Dr. Anurag Agrawal (PI)
   - Site PIs: Dr. Jeba, Dr. Ghoshal, Dr. Warrier, Dr. Kannan
   - Site coordinators (4 people)
5. Click **Review release** > **Start rollout to Internal testing**
6. Share the opt-in link with testers

### 10.3 Set Up Closed Testing Track

1. Go to **Release > Testing > Closed testing**
2. Create a track named "EVAH Evaluation Sites"
3. Add testers:
   - Create a Google Group per site: `palli-sahayak-cmc@googlegroups.com`, etc.
   - Add site coordinators and initial ASHA worker participants
   - Each group can have up to 2,000 members
4. Upload the AAB
5. **Review release** > **Start rollout to Closed testing**

### 10.4 Pre-Launch Report

After uploading to any track, Google automatically runs:
- **Accessibility**: Checks for touch target sizes, contrast ratios
- **Security**: Scans for vulnerabilities
- **Performance**: Tests on reference devices
- **Compatibility**: Tests across Android versions

Review the pre-launch report at **Release > Pre-launch report** and fix any critical issues before production release.

---

## 11. Production Release

### 11.1 Pre-Release Checklist

- [ ] Internal testing completed with no critical bugs
- [ ] Closed testing at all 4 sites completed
- [ ] Pre-launch report issues addressed
- [ ] Store listing complete (description, screenshots, icon)
- [ ] Content rating completed
- [ ] Data safety form completed
- [ ] Pricing and distribution set
- [ ] All policy declarations accepted
- [ ] `versionCode` incremented from testing builds

### 11.2 Release to Production

1. Go to **Release > Production**
2. Click **Create new release**
3. Upload the final AAB (or promote from closed testing)
4. Add release notes
5. Set **Rollout percentage**: Start with **20%** for safety
6. Click **Review release** > **Start rollout to Production**

### 11.3 Staged Rollout

| Day | Rollout % | Action |
|-----|-----------|--------|
| Day 1-3 | 20% | Monitor crash rate, ANR rate, uninstall rate |
| Day 4-7 | 50% | If metrics stable, increase |
| Day 8-14 | 100% | Full rollout if no issues |

Monitor at **Quality > Android Vitals**:
- **Crash rate**: Target < 1%
- **ANR rate**: Target < 0.5%
- **Excessive wakeups**: Target < 10/hour

### 11.4 Rollback Procedure

If critical issues found after production release:
1. Go to **Release > Production**
2. Click **Halt rollout** (stops serving to new users)
3. Fix the issue
4. Upload new AAB with incremented `versionCode`
5. Resume rollout

---

## 12. Post-Launch Checklist

### 12.1 Monitoring

| What | Where | Frequency |
|------|-------|-----------|
| Crash reports | Firebase Crashlytics / Play Console | Daily for first 2 weeks |
| Android Vitals | Play Console > Quality | Daily for first 2 weeks |
| User ratings/reviews | Play Console > Ratings | Daily |
| Install/uninstall trends | Play Console > Statistics | Weekly |
| Backend API errors | Server logs (`rag_server.log`) | Daily |

### 12.2 Update Cadence

| Update Type | Frequency | When |
|-------------|-----------|------|
| Bug fixes | As needed | Within 48 hours of critical bug discovery |
| Minor features | Bi-weekly | During EVAH active evaluation (Months 5-8) |
| Major releases | Monthly | Aligned with EVAH phases |

### 12.3 Version Numbering

```
versionName: MAJOR.MINOR.PATCH
versionCode: always incrementing integer

Examples:
0.1.0 (versionCode 1)  — Initial release
0.1.1 (versionCode 2)  — Bug fix
0.2.0 (versionCode 3)  — New feature (e.g., additional language)
1.0.0 (versionCode 10) — Stable release after EVAH Phase 2
```

### 12.4 Update Process

```bash
# 1. Increment versionCode and versionName in app/build.gradle.kts
# 2. Build release bundle
./gradlew bundleAllSitesRelease

# 3. Upload to Play Console
# 4. Write release notes
# 5. Start staged rollout (20% -> 50% -> 100%)
```

---

## 13. EVAH Evaluation-Specific Notes

### 13.1 Deployment Timeline per EVAH Phases

| EVAH Phase | Months | Play Store Action |
|------------|--------|-------------------|
| Phase 1 (Preparation) | 1-2 | Internal testing with site coordinators |
| Phase 2 (Training) | 3-4 | Closed testing at 4 sites, supervised deployment |
| Phase 3 (Active Evaluation) | 5-8 | Production release to all 200 participants |
| Phase 4 (Analysis) | 9-10 | Maintenance updates only |

### 13.2 Site-Specific APK Distribution

For CCHRC Silchar (remote, limited Play Store access):
1. Build site-specific APK: `./gradlew assembleCchrcSilcharRelease`
2. Transfer APK to site coordinator via secure file sharing
3. Site coordinator installs on participant devices via USB or local file share
4. Enable "Install from unknown sources" on participant devices

### 13.3 Participant Onboarding via Play Store

For sites with Play Store access (CMC Vellore, KMC Manipal, CCF Coimbatore):
1. Share the Play Store link via the closed testing track
2. Each participant:
   a. Opens the link on their Android phone
   b. Taps "Install" on the Play Store page
   c. Opens the app and completes onboarding (language, role, PIN)
   d. Site coordinator verifies successful setup

### 13.4 Data Collection Compliance

Ensure the Play Store listing mentions:
- The app collects evaluation data as part of the EVAH research study
- Participants have provided informed consent
- Data is used for research purposes only
- All data handling complies with DPDP Act 2023

### 13.5 Evaluation Builds vs Production Builds

Consider using build flavors to distinguish:
- **Evaluation builds** (closed testing): All evaluation instrumentation enabled (SUS, vignettes, time-motion)
- **Production builds** (post-evaluation): Evaluation instrumentation can be disabled via config flag

```kotlin
// In app/build.gradle.kts
productFlavors {
    create("evaluation") {
        buildConfigField("Boolean", "EVALUATION_MODE", "true")
    }
    create("production") {
        buildConfigField("Boolean", "EVALUATION_MODE", "false")
    }
}
```

---

## Quick Reference: Key Commands

```bash
# Build debug APK (for testing)
./gradlew assembleAllSitesDebug

# Build release AAB (for Play Store)
./gradlew bundleAllSitesRelease

# Build release APK (for direct distribution)
./gradlew assembleAllSitesRelease

# Build site-specific APK
./gradlew assembleCmcVelloreRelease
./gradlew assembleKmcManipalRelease
./gradlew assembleCcfCoimbatoreRelease
./gradlew assembleCchrcSilcharRelease

# Run unit tests
./gradlew test

# Run lint
./gradlew lint

# Check APK size
ls -lh app/build/outputs/apk/allSites/release/app-allSites-release.apk
```

---

## Quick Reference: Play Console URLs

| Page | URL |
|------|-----|
| Play Console Home | https://play.google.com/console |
| Create New App | Play Console > All apps > Create app |
| Internal Testing | Release > Testing > Internal testing |
| Closed Testing | Release > Testing > Closed testing |
| Production Release | Release > Production |
| Android Vitals | Quality > Android vitals |
| Pre-Launch Report | Release > Pre-launch report |
| Store Listing | Grow > Store presence > Main store listing |
| Content Rating | Policy > App content > Content rating |
| Data Safety | Policy > App content > Data safety |

---

**End of Document**

*Palli Sahayak Google Play Store Publishing Guide | 27 March 2026*
