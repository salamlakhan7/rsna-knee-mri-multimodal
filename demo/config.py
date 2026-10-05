import os
BASE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "KneeScope"
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", os.path.join(BASE, "models", "resnet18_attn.pt"))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COVERED_LANGS = {"en", "es", "de", "tr"}
# Measured on the 58 gold-labeled studies (see Results page)
IMAGE_AUC = {"ACL": .69, "MCL": .63, "Medial Meniscus": .60, "Lateral Meniscus": .71, "Medial OA": .76,
             "Lateral OA": .85, "PF OA": .66, "Effusion": .70, "Synovitis": .61, "Baker's": .78,
             "Contusion": .67, "Fracture": .63}
RULE_F1 = {"ACL": .92, "MCL": .67, "Medial Meniscus": .85, "Lateral Meniscus": .89, "Medial OA": .67,
           "Lateral OA": .63, "PF OA": .43, "Effusion": .73, "Synovitis": .61, "Baker's": .71,
           "Contusion": .63, "Fracture": .61}
CONTACT = {"name": "Abdul Salam", "email": "salamlakhan7@gmail.com",
           "github": "https://github.com/salamlakhan7", "linkedin": ""}
DISCLAIMER = ("Research and learning project. Outputs are not a diagnosis and must not be used for clinical decisions.")
SAMPLES = {
    "English": {"findings": ["ACL", "Medial Meniscus", "Effusion"],
                "text": "Technique: MRI of the right knee. Findings: Complete tear of the anterior cruciate ligament. "
                        "Horizontal tear of the posterior horn of the medial meniscus. Moderate joint effusion. "
                        "The lateral meniscus is intact. No fracture is seen."},
    "Spanish": {"findings": ["ACL", "Effusion", "Contusion", "Baker's"],
                "text": "Técnica: RMN de rodilla. Resultados: Rotura del ligamento cruzado anterior. Derrame articular. "
                        "Contusión ósea en cóndilo femoral lateral. Menisco interno sin signos de rotura. Quiste de Baker."},
    "German": {"findings": ["ACL", "Contusion"],
               "text": "Technik: MRT des linken Kniegelenks. Befund: Riss des vorderen Kreuzbandes. Kein Gelenkerguss. "
                       "Knochenmarködem im lateralen Tibiaplateau. Innenmeniskus intakt."},
    "Turkish": {"findings": ["ACL", "PF OA"],
                "text": "Teknik: Diz MRG. Bulgular: Ön çapraz bağda yırtık izlenmiştir. Eklem efüzyonu izlenmemiştir. "
                        "Patellofemoral eklem dejenerasyonu ile uyumlu fokal kıkırdak kayıpları mevcuttur."},
}
def reliability(auc):
        return "🟢 more reliable" if auc >= 0.75 else ("🟠 moderate" if auc >= 0.65 else "⚪ low")
