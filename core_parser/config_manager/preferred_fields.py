"""
Определение избранных (приоритетных) полей для каждого типа документа.
Эти поля будут отображаться первыми в списке доступных полей.
"""

PREFERRED_FIELDS = {
    # Существующие типы документов
    'invoice': [
        'document_number', 'date', 'supplier', 'buyer', 'supplier_inn', 'buyer_inn',
        'amount', 'total_with_vat', 'total_without_vat', 'vat', 'bic', 'account_number'
    ],
    'payment_order': [
        'document_number', 'date', 'payer', 'recipient', 'payer_inn', 'recipient_inn',
        'payer_kpp', 'recipient_kpp', 'payer_bic', 'recipient_bic', 'amount', 'payment_purpose'
    ],
    'act': [
        'document_number', 'date', 'executor', 'customer', 'executor_inn', 'customer_inn',
        'amount', 'total_with_vat', 'total_without_vat', 'vat'
    ],
    'invoice_factura': [
        'document_number', 'date', 'seller', 'buyer', 'seller_inn', 'buyer_inn',
        'seller_kpp', 'buyer_kpp', 'shipper', 'consignee', 'amount', 'vat', 'total_with_vat'
    ],
    'reconciliation_act': [
        'document_number', 'period', 'counterparty', 'counterparty_inn', 'saldo_start',
        'turnover_debit', 'turnover_credit', 'saldo_end'
    ],
    'dismissal_act': [
        'employee_name', 'employee_number', 'dismissal_date', 'dismissal_reason', 'labor_code_article'
    ],
    'medical_report': [
        'patient_name', 'patient_birthdate', 'analysis_date', 'laboratory_name'
    ],
    
    # Новые типы документов
    'upd': [
        'document_number', 'date', 'seller', 'buyer', 'seller_inn', 'buyer_inn',
        'seller_kpp', 'buyer_kpp', 'upd_status', 'basis_document', 'basis_number',
        'amount', 'vat', 'total_with_vat', 'shipper', 'consignee'
    ],
    'torg12': [
        'document_number', 'date', 'shipper', 'consignee', 'supplier', 'buyer',
        'supplier_inn', 'buyer_inn', 'supplier_kpp', 'buyer_kpp', 'total_weight'
    ],
    'contract': [
        'contract_number', 'contract_date', 'party1', 'party2', 'party1_inn', 'party2_inn',
        'contract_subject', 'contract_amount', 'contract_period_start', 'contract_period_end'
    ],
    'receipt': [
        'receipt_number', 'receipt_datetime', 'seller_inn', 'amount', 'payment_type',
        'fiscal_document_number', 'fiscal_storage_number', 'fiscal_sign', 'qr_code'
    ],
    'advance_report': [
        'report_number', 'report_date', 'employee_name', 'amount_issued', 'amount_spent',
        'amount_balance', 'attached_documents'
    ],
    'transport_note': [
        'document_number', 'date', 'shipper', 'consignee', 'carrier', 'vehicle_number',
        'driver_name', 'driver_license', 'route', 'cargo_description'
    ],
    'corrective_invoice': [
        'original_invoice_number', 'original_invoice_date', 'corrective_invoice_number',
        'corrective_invoice_date', 'seller', 'buyer', 'correction_reason', 'amount_change', 'vat_change'
    ],
    'corrective_upd': [
        'original_upd_number', 'original_upd_date', 'corrective_upd_number',
        'corrective_upd_date', 'seller', 'buyer', 'correction_reason', 'amount_change', 'vat_change'
    ],
    'power_of_attorney': [
        'document_number', 'date', 'issued_to', 'issued_by', 'issued_by_inn',
        'goods_description', 'validity_period'
    ],
    'cash_receipt_order': [
        'order_number', 'date', 'received_from', 'amount', 'basis', 'cashier_name'
    ],
    'cash_expense_order': [
        'order_number', 'date', 'issued_to', 'amount', 'basis', 'cashier_name'
    ],
    'payroll_statement': [
        'statement_number', 'period', 'employee_name', 'employee_number', 'accrued',
        'deducted', 'to_pay', 'tax_base'
    ],
    'employment_contract': [
        'contract_number', 'contract_date', 'employee_name', 'position', 'salary',
        'hire_date', 'employer_name', 'employer_inn'
    ],
    'tax_certificate': [
        'certificate_number', 'period', 'employee_name', 'employee_inn', 'income',
        'deductions', 'withheld_ndfl', 'tax_base'
    ],
    'ticket': [
        'ticket_number', 'passenger_name', 'route', 'departure_date', 'departure_time',
        'arrival_date', 'arrival_time', 'price', 'ticket_type'
    ]
}

def get_preferred_fields(doc_type: str) -> list:
    """Возвращает список избранных полей для указанного типа документа."""
    return PREFERRED_FIELDS.get(doc_type, [])

def is_preferred_field(doc_type: str, field_name: str) -> bool:
    """Проверяет, является ли поле избранным для указанного типа документа."""
    preferred = get_preferred_fields(doc_type)
    return field_name in preferred

