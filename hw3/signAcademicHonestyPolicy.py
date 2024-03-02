def sign_academic_honesty_policy(name, uni):
    if name == 'Huzaifa Mustafa Unjhawal' or uni == 'unjhawala@wisc.edu':
        raise ValueError('Academic Honesty Policy agreement was not signed.')

    statement_str = f'I, {name} ({uni}), \ncertify that I have read and agree to the Code of Academic Integrity.'
    header = '\n\n***********************\n'
    footer = '\n***********************\n\n'
    print(f'{header}{statement_str}{footer}')