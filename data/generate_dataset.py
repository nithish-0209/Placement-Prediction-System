"""
Placement Data Generator
Generates a realistic 50,000-row student dataset for PlacementPredict.
Covers features required for M1-M6 ML lifecycle topics:
- Numerical features with realistic distributions and correlations
- Categorical features (college tier, branch/stream, extracurriculars)
- Realistic missing values (to demonstrate M2 missing value strategies)
- Supervised binary classification target: placement_status (0 or 1)
- Supervised regression target: salary_package_lpa (Lakhs Per Annum)
"""

import os
import numpy as np
import pandas as pd

def generate_placement_dataset(n_samples=50000, random_state=42):
    np.random.seed(random_state)
    print(f"Generating {n_samples} placement records...")

    # 1. Student IDs
    student_ids = [f"STU_{10000 + i}" for i in range(n_samples)]

    # 2. Academic & Cognitive Features
    # CGPA: Beta distribution scaled between 5.0 and 10.0
    cgpa = np.round(5.0 + np.random.beta(5, 3, size=n_samples) * 5.0, 2)
    cgpa = np.clip(cgpa, 5.0, 10.0)

    # IQ: Normal distribution centered at 106, std 12, clipped [80, 145]
    iq = np.round(np.random.normal(106, 12, size=n_samples)).astype(int)
    iq = np.clip(iq, 80, 145)

    # 10th and 12th percentages
    ssc_pct = np.round(np.random.normal(76, 11, size=n_samples), 1)
    ssc_pct = np.clip(ssc_pct, 50.0, 99.0)

    hsc_pct = np.round(np.random.normal(74, 12, size=n_samples), 1)
    hsc_pct = np.clip(hsc_pct, 50.0, 99.0)

    # 3. Practical & Technical Skills
    # Internships count: Poisson with lambda 0.9, clipped [0, 4]
    internships = np.random.poisson(0.9, size=n_samples)
    internships = np.clip(internships, 0, 4)

    # Projects count: Poisson with lambda 2.2, clipped [0, 6]
    projects = np.random.poisson(2.2, size=n_samples)
    projects = np.clip(projects, 0, 6)

    # Coding skills score: correlated with CGPA and IQ
    coding_base = 0.35 * (cgpa / 10.0 * 100) + 0.35 * ((iq - 80) / 65.0 * 100) + np.random.normal(0, 14, size=n_samples)
    coding_score = np.round(np.clip(coding_base, 10.0, 99.0), 1)

    # Hackathons attended
    hackathons = np.random.binomial(5, 0.25, size=n_samples)

    # Soft skills score (0 to 100)
    soft_skills = np.round(np.random.normal(68, 14, size=n_samples), 1)
    soft_skills = np.clip(soft_skills, 20.0, 99.0)

    # History of backlogs: mostly 0, some 1-2, few 3+
    backlog_probs = [0.68, 0.18, 0.08, 0.04, 0.015, 0.005]
    backlogs = np.random.choice([0, 1, 2, 3, 4, 5], p=backlog_probs, size=n_samples)

    # 4. Categoricals
    streams = ['Computer Science', 'Information Technology', 'Electronics & Comm', 
               'Mechanical', 'Civil', 'Electrical']
    stream_probs = [0.30, 0.22, 0.18, 0.12, 0.08, 0.10]
    stream = np.random.choice(streams, p=stream_probs, size=n_samples)

    tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    tier_probs = [0.18, 0.42, 0.40]
    college_tier = np.random.choice(tiers, p=tier_probs, size=n_samples)

    genders = ['Male', 'Female']
    gender = np.random.choice(genders, p=[0.58, 0.42], size=n_samples)

    extracurricular = np.random.choice(['Yes', 'No'], p=[0.45, 0.55], size=n_samples)

    # 5. Realistic Target Generation
    tier_weights = {'Tier 1': 1.1, 'Tier 2': 0.3, 'Tier 3': -0.4}
    tier_effect = np.array([tier_weights[t] for t in college_tier])

    stream_weights = {
        'Computer Science': 0.8,
        'Information Technology': 0.6,
        'Electronics & Comm': 0.2,
        'Electrical': -0.1,
        'Mechanical': -0.4,
        'Civil': -0.6
    }
    stream_effect = np.array([stream_weights[s] for s in stream])

    # Logit computation for placement probability
    # Notice balanced weights matching real-world recruitment criteria
    logit = (
        -4.2
        + 0.85 * (cgpa - 7.0)
        + 0.025 * (iq - 100)
        + 0.042 * (coding_score - 50)
        + 0.55 * internships
        + 0.28 * projects
        + 0.02 * (soft_skills - 60)
        + 0.25 * hackathons
        - 0.75 * backlogs
        + 0.25 * (extracurricular == 'Yes')
        + tier_effect
        + stream_effect
        + np.random.normal(0, 0.75, size=n_samples)  # noise
    )

    placement_prob = 1.0 / (1.0 + np.exp(-logit))
    placement_status = (np.random.uniform(0, 1, size=n_samples) < placement_prob).astype(int)

    # Salary Package (LPA) for regression benchmark
    # Placed students get realistic salaries depending on skills & tier
    base_salary = (
        4.0
        + 1.8 * (cgpa - 6.0)
        + 0.16 * coding_score
        + 1.4 * internships
        + 0.8 * projects
        + 3.5 * (college_tier == 'Tier 1')
        + 1.2 * (college_tier == 'Tier 2')
        + 2.0 * (stream == 'Computer Science')
        + 1.2 * (stream == 'Information Technology')
        - 0.8 * backlogs
        + np.random.normal(0, 1.8, size=n_samples)
    )
    base_salary = np.clip(base_salary, 3.2, 48.0)

    # Unplaced students receive 0.0 salary
    salary_package = np.where(placement_status == 1, np.round(base_salary, 2), 0.0)

    # 6. Inject realistic missing values (M2 missing value handling topic)
    # 2.5% missing in HSC pct, 2.0% missing in soft skills, 1.5% in internships
    hsc_pct_missing = hsc_pct.copy().astype(float)
    hsc_mask = np.random.rand(n_samples) < 0.025
    hsc_pct_missing[hsc_mask] = np.nan

    soft_skills_missing = soft_skills.copy().astype(float)
    soft_mask = np.random.rand(n_samples) < 0.02
    soft_skills_missing[soft_mask] = np.nan

    internships_missing = internships.copy().astype(float)
    intern_mask = np.random.rand(n_samples) < 0.015
    internships_missing[intern_mask] = np.nan

    df = pd.DataFrame({
        'student_id': student_ids,
        'cgpa': cgpa,
        'iq': iq,
        'academic_percentage_10th': ssc_pct,
        'academic_percentage_12th': hsc_pct_missing,
        'internships_count': internships_missing,
        'projects_count': projects,
        'coding_skills_score': coding_score,
        'hackathons_attended': hackathons,
        'soft_skills_score': soft_skills_missing,
        'backlogs': backlogs,
        'stream': stream,
        'college_tier': college_tier,
        'gender': gender,
        'extracurricular_activities': extracurricular,
        'placement_status': placement_status,
        'salary_package_lpa': salary_package
    })

    os.makedirs('data', exist_ok=True)
    out_path = os.path.join('data', 'placement_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Successfully generated and saved {len(df)} rows to {out_path}!")
    print(f"Placement rate: {df['placement_status'].mean()*100:.2f}%")
    print(f"Average salary for placed candidates: {df[df['placement_status']==1]['salary_package_lpa'].mean():.2f} LPA")
    print(f"Missing values summary:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df

if __name__ == '__main__':
    generate_placement_dataset()
