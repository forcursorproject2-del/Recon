import os
from ruamel.yaml import YAML
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        yaml = YAML()
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.load(f)
        else:
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'document_signatures': {
                'invoice': {
                    'keywords': ['счет', 'инвойс', 'счет-фактура', 'сумма', 'налог'],
                    'patterns': [r'счет\s*№?\s*\d+', r'инвойс\s*№?\s*\d+', r'счет-фактура\s*№?\s*\d+'],
                    'exclude': ['платежное', 'поручение']
                },
                'payment_order': {
                    'keywords': ['платежное', 'поручение', 'сумма', 'получатель', 'банковские'],
                    'patterns': [r'платежное\s*поручение\s*№?\s*\d+', r'платежное\s*поручение'],
                    'exclude': ['счет', 'инвойс']
                },
                'act': {
                    'keywords': ['акт', 'приемки', 'работ', 'услуг'],
                    'patterns': [r'акт\s*приемки\s*работ', r'акт\s*оказания\s*услуг'],
                    'exclude': ['сверки', 'взаиморасчетов']
                },
                'invoice_factura': {
                    'keywords': ['счет-фактура', 'налог', 'ндс', 'продавец', 'покупатель'],
                    'patterns': [r'счет-фактура\s*№?\s*\d+', r'счет-фактура'],
                    'exclude': ['платежное']
                },
                'reconciliation_act': {
                    'keywords': ['акт сверки', 'взаиморасчеты', 'сальдо', 'обороты', 'дебет', 'кредит'],
                    'patterns': [r'акт\sсверки\s№?\s*\d+', r'акт\s*взаиморасчетов'],
                    'exclude': ['счет', 'платежное']
                }
            },
            'field_patterns': {
                'inn': {
                    'pattern': r'ИНН\s*[:\-]?\s*(\d{10}|\d{12})',
                    'validate': 'digits_10_12'
                },
                'kpp': {
                    'pattern': r'КПП\s*[:\-]?\s*(\d{9})',
                    'validate': 'digits_9'
                },
                'amount': {
                    'pattern': r'сумма\s*[:\-]?\s*([\d\s,]+\.?\d*)',
                    'validate': 'float'
                },
                'date': {
                    'pattern': r'дата\s*[:\-]?\s*(\d{2}\.\d{2}\.\d{4})',
                    'validate': 'date'
                },
                'saldo_start': {
                    'pattern': r'сальдо\s*на\s*начало\s*[:\-]?\s*([\d\s,]+\.?\d*)',
                    'validate': 'float'
                },
                'turnover_debit': {
                    'pattern': r'оборот\s*по\s*дебету\s*[:\-]?\s*([\d\s,]+\.?\d*)',
                    'validate': 'float'
                },
                'turnover_credit': {
                    'pattern': r'оборот\s*по\s*кредиту\s*[:\-]?\s*([\d\s,]+\.?\d*)',
                    'validate': 'float'
                },
                'saldo_end': {
                    'pattern': r'сальдо\s*на\s*конец\s*[:\-]?\s*([\d\s,]+\.?\d*)',
                    'validate': 'float'
                }
            }
        }

    def get_signatures(self) -> Dict[str, Any]:
        return self.config.get('document_signatures', {})

    def get_patterns(self) -> Dict[str, Any]:
        return self.config.get('field_patterns', {})

    def get_classifier_mode(self) -> str:
        return self.config.get("classifier", {}).get("mode", "rules_only")

    def use_ml(self) -> bool:
        return self.config.get("classifier", {}).get("use_ml", False)

    def use_bert(self) -> bool:
        return self.config.get("classifier", {}).get("use_bert", False)
