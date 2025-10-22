crr_cols_mapping = {
    'Reb Number': 'reb_number',
    'REB Approval Date': 'reb_approval_date',
    # '': 'rsh_study_code',
    'Study Number': 'study_name',
    'Title': 'study_title',
    # '': 'regulated_study',
    'Status': 'study_status',
    'Study Link': 'study_link',
    'Study Activation Date': 'study_activation_date',
    'Department': 'department',
    'Sponsor Type': 'sponsor_type',
    'Investigator Initiated': 'investigator_initiated',
    'Primary Investigator': 'primary_investigator',
    'Phase': 'phase',
    'CTC AE Version': 'ctcae_version',

    # '': 'patient_name',
    'MRN': 'mrn',
    'Birthdate': 'birthdate',
    'Age at Enrolment': 'age_at_enrolment',
    'Enrolled Date/Time': 'enrolled_date',
    'Disease Site Group': 'disease_site_group',
    'Disease Location': 'disease_location',

    '(AE) Category': 'ae_category',
    '(AE) Term Define': 'ae_term',
    '(AE) Specific': 'ae_specific',
    # '': 'ae_status',
    '(AE) Treatment Required': 'ae_treatment_required',
    '(AE) SAE': 'sae',

    '(AE) Grade': 'ae_grade',
    'Start Date': 'ae_grade_start_date',
     'End Date': 'ae_grade_end_date',
    
    '(AE) Start Date Day': 'ae_grade_start_date_day',
    '(AE) Start Date Month': 'ae_grade_start_date_month',
    '(AE) Start Date Year': 'ae_grade_start_date_year',
    '(AE) End Date Month': 'ae_grade_end_date_month',
    '(AE) End Date Day': 'ae_grade_end_date_day',
    '(AE) End Date Year': 'ae_grade_end_date_year',
}



epic_cols_mapping = {
    # '': 'reb_number',
    # '': 'reb_approval_date',
    'RSH Study Code': 'rsh_study_code',
    'Study Name': 'study_name',
    'Study Title': 'study_title',
    'Regulated Study': 'regulated_study',
    # '': 'study_status',
    # '': 'study_link',
    # '': 'study_activation_date',
    # '': 'department',
    'Academic/Industry Study': 'sponsor_type',
    'Investigator Initiated': 'investigator_initiated',
    'Principal Investigator': 'primary_investigator',
    'CCRU: Phase': 'phase',
    # '': 'ctcae_version',

    'Patient Name': 'patient_name',
    'MRN': 'mrn',
    # '': 'birthdate',
    # '': 'age_at_enrolment',
    # '': 'enrolled_date',
    'Disease Site Group': 'disease_site_group',
    # '': 'disease_location',

    # '': 'ae_category',
    'Event Name': 'ae_term',
    # '': 'ae_specific',
    'Status': 'ae_status',
    # '': 'ae_treatment_required',
    # '': 'sae',

    'Grade Hx': 'ae_grade',
    'Grade Start Hx': 'ae_grade_start_date',
    
    'Event Start Dt': 'ae_start_date',
    'Resolved Date': 'ae_end_date',
    'Current Grade': 'ae_current_grade',
    'Cur Grade Start Dt': 'ae_current_grade_start_date',
    'Highest Grade': 'ae_highest_grade',
}


cols_of_interest = [
    'study_name',
    'study_title',
    'sponsor_type',
    'investigator_initiated',
    'primary_investigator',
    'phase',
    'detected_ctcae_version',
    'mrn',
    'birthdate',
    'age_at_enrolment',
    'disease_site_group',
    'ae_category',
    'ae_term',
    'ae_specific',
    'ae_grade',
    'ae_grade_start_date',
    'mapped_soc',
    'mapped_term',
    'mapped_grade'
]

ae_term_corrections = {
    'Gastoparesis': 'Gastroparesis',
    'Musculoskeletal and connective tissue disorder -  Other, specify': 'Musculoskeletal and connective tissue disorder - Other, specify',
    'Heart Failure': 'Heart failure',
    'Atrial Flutter': 'Atrial flutter',
    'Chest Wall Pain': 'Chest wall pain'
}

month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'Novemeber': 11, 'December': 12
}