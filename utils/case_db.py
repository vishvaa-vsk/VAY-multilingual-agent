import re

MOCK_CASES = {
    "CASE-1092": {
        "title": "Auto Insurance Collision Claim",
        "category": "Insurance",
        "status": "Under Active Review",
        "details": "Adjuster report submitted. Settlement calculation in progress.",
        "eta": "2 business days",
        "responses": {
            "en": "Certainly. I am looking into your case, CASE-1092. Your auto collision claim is under active review with an estimated resolution within two business days.",
            "es": "Por supuesto. Estoy revisando su caso, CASE-1092. Su reclamo de seguro de automóvil está en revisión activa con una resolución estimada en dos días hábiles.",
            "fr": "Certainement. Je consulte votre dossier, CASE-1092. Votre réclamation d'assurance automobile est en cours d'examen actif, résolution prévue sous deux jours ouvrables.",
            "de": "Gewiss. Ich überprüfe Ihren Fall, CASE-1092. Ihre Kfz-Schadensmeldung befindet sich in der aktiven Prüfung. Voraussichtliche Bearbeitung: zwei Werktage.",
            "hi": "निश्चय ही। मैं आपके केस CASE-1092 को देख रहा हूँ। आपका ऑटो बीमा दावा सक्रिय समीक्षाधीन है।",
            "ja": "承知いたしました。ケースCASE-1092を確認しております。自動車保険の補償請求は審査中であり、2営業日以内に完了する見込みです。"
        }
    },
    "CASE-8834": {
        "title": "Superior Court Legal Appeal",
        "category": "Legal",
        "status": "Hearing Docketed",
        "details": "Briefs filed by appellate council. Scheduled for oral argument before Panel B.",
        "eta": "September 15, 2026",
        "responses": {
            "en": "I have retrieved CASE-8834. Your legal appeal hearing has been official docketed for oral argument on September 15th before Panel B.",
            "es": "He recuperado el caso CASE-8834. La audiencia de su apelación legal ha sido programada oficialmente para el 15 de septiembre ante el Panel B.",
            "fr": "J'ai extrait le dossier CASE-8834. L'audience de votre appel juridique est fixée au 15 septembre devant la Chambre B.",
            "de": "Ich habe den Fall CASE-8834 abgerufen. Ihre Berufungsverhandlung ist für den 15. September vor Panel B angesetzt.",
            "hi": "मैंने आपका केस CASE-8834 प्राप्त कर लिया है। आपकी कानूनी अपील की सुनवाई 15 सितंबर को निर्धारित की गई है।",
            "ja": "ケースCASE-8834を取得しました。法的控訴審判は9月15日に第B審判部にて開廷予定です。"
        }
    },
    "CASE-4021": {
        "title": "Medical Benefits Reimbursement",
        "category": "Healthcare",
        "status": "Approved for Disbursement",
        "details": "Claim verified against policy terms. Transfer queued to registered bank account.",
        "eta": "Disbursement pending ($4,250)",
        "responses": {
            "en": "Looking into CASE-4021. Good news: your medical coverage claim of $4,250 has been fully approved and queued for direct deposit.",
            "es": "Revisando el caso CASE-4021. Buenas noticias: su reclamo de cobertura médica por $4,250 ha sido aprobado y puesto en cola para depósito directo.",
            "fr": "Consultation du dossier CASE-4021. Bonne nouvelle: votre demande de remboursement médical de 4 250 $ a été approuvée et mise en paiement.",
            "de": "Überprüfe Fall CASE-4021. Gute Nachrichten: Ihren medizinischen Erstattungsanspruch über 4.250 $ haben wir vollständig genehmigt.",
            "hi": "केस CASE-4021 की जाँच की गई। अच्छी खबर है: आपका $4,250 का चिकित्सा दावा पूरी तरह से स्वीकृत हो गया है।",
            "ja": "ケースCASE-4021を確認しました。朗報です。4,250ドルの医療給付請求が承認され、口座振込の手続きに入りました。"
        }
    },
    "CASE-7710": {
        "title": "Global Talent Visa Renewal",
        "category": "Immigration",
        "status": "Biometrics Verified",
        "details": "Security clearance background checks completed. Pending consular signature.",
        "eta": "5 to 7 days",
        "responses": {
            "en": "Accessing CASE-7710. Biometric verification for your visa renewal is complete. Final consular signature is pending.",
            "es": "Accediendo al caso CASE-7710. La verificación biométrica para la renovación de su visa está completa. Pendiente de firma consular final.",
            "fr": "Accès au dossier CASE-7710. La vérification biométrique pour votre renouvellement de visa est terminée. En attente de signature consulaire.",
            "de": "Zugriff auf Fall CASE-7710. Die biometrische Verifizierung für Ihre Visumverlängerung ist abgeschlossen. Es fehlt nur noch die konsularische Unterschrift.",
            "hi": "केस CASE-7710 एक्सेस किया जा रहा है। आपके वीजा नवीनीकरण के लिए बायोमेट्रिक सत्यापन पूरा हो गया है।",
            "ja": "ケースCASE-7710にアクセスしました。ビザ更新の生体認証確認が完了し、領事署名待ちです。"
        }
    }
}

DEFAULT_PROMPT_RESPONSES = {
    "en": "Certainly, I am looking into your case. Please provide your case ID.",
    "es": "Ciertamente, estoy revisando su caso. Por favor proporcione su ID de caso.",
    "fr": "Certainement, je consulte votre dossier. Veuillez me fournir votre identifiant de dossier.",
    "de": "Gewiss, ich überprüfe Ihren Fall. Bitte geben Sie Ihre Fall-ID an.",
    "hi": "निश्चय ही, मैं आपके मामले की जाँच कर रहा हूँ। कृपया अपनी केस आईडी प्रदान करें।",
    "ja": "承知いたしました。ケースを確認しております。ケースIDをお知らせください。"
}

def process_query(transcript: str, lang: str = "en") -> str:
    """
    Process a user's vocal/text transcript and return the assistant response.
    """
    if not transcript or not transcript.strip():
        return DEFAULT_PROMPT_RESPONSES.get(lang, DEFAULT_PROMPT_RESPONSES["en"])

    text = transcript.upper()
    
    # Check for specific case ID regex match: e.g. CASE-1092, 1092, 8834, 4021, 7710
    match = re.search(r'(CASE[-_\s]?)?(\d{4})', text)
    if match:
        num = match.group(2)
        full_id = f"CASE-{num}"
        if full_id in MOCK_CASES:
            case_info = MOCK_CASES[full_id]
            return case_info["responses"].get(lang, case_info["responses"]["en"])

    # Generic search by keyword
    for case_id, info in MOCK_CASES.items():
        if case_id in text or info["title"].upper() in text or info["category"].upper() in text:
            return info["responses"].get(lang, info["responses"]["en"])

    if "STATUS" in text or "LOOKUP" in text or "CHECK" in text or "CASE" in text:
        return DEFAULT_PROMPT_RESPONSES.get(lang, DEFAULT_PROMPT_RESPONSES["en"])

    # Fallback polite prompt requesting Case ID
    if lang == "es":
        return f"He recibido su mensaje: \"{transcript}\". Para acceder a los detalles específicos, por favor mencione su número de caso (por ejemplo, CASE-1092)."
    elif lang == "fr":
        return f"J'ai bien reçu votre demande: «{transcript}». Pour accéder à votre dossier, veuillez indiquer votre numéro de dossier (ex: CASE-1092)."
    elif lang == "de":
        return f"Ich habe Ihre Anfrage erhalten: „{transcript}“. Um Fortzufahren, nennen Sie bitte Ihre Fall-ID (z. B. CASE-1092)."
    elif lang == "hi":
        return f"मुझे आपका संदेश प्राप्त हुआ: \"{transcript}\"। विशिष्ट विवरण के लिए कृपया अपना केस आईडी (जैसे CASE-1092) प्रदान करें।"
    elif lang == "ja":
        return f"リクエストを受信しました: 「{transcript}」。詳細を確認するため、ケースID（例: CASE-1092）をお知らせください。"
    else:
        return f"I heard: \"{transcript}\". To pull up your records, please speak or enter your 4-digit case ID (such as CASE-1092)."
