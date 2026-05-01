import os
import django
from pathlib import Path

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from apps.simulations.models import Simulation
from django.conf import settings

# Paths to generated images (absolute paths from task history)
IMAGE_DIR = Path("/home/cardan/.gemini/antigravity/brain/429c68f3-a913-4226-a5c7-e147e895be6b/")
MEDIA_SIM_DIR = Path(settings.MEDIA_ROOT) / "simulations"
MEDIA_SIM_DIR.mkdir(parents=True, exist_ok=True)

# Helper to copy image to media
def setup_image(image_basename):
    # Find the most recent file matching the pattern
    matches = list(IMAGE_DIR.glob(f"{image_basename}_*.png"))
    if not matches:
        return ""
    latest_image = max(matches, key=os.path.getctime)
    dest = MEDIA_SIM_DIR / f"{image_basename}.png"
    import shutil
    shutil.copy(latest_image, dest)
    return f"{settings.MEDIA_URL}simulations/{image_basename}.png"

NEW_SIMULATIONS = [
    {
        "name": "Dakar Tech Hub Networking",
        "description": "Network with tech leaders at the Dakar Digital Hub. Practice your elevator pitch and professional small talk.",
        "category": "networking",
        "difficulty": "beginner",
        "icon_emoji": "🤝",
        "image_key": "networking_dakar_tech_hub",
        "system_prompt": "You are a senior developer and a startup founder at a networking event in Dakar. You are looking for collaborators. Be friendly but professional."
    },
    {
        "name": "Abidjan FinTech Pitch",
        "description": "Pitch your innovative financial solution to a panel of demanding investors in the Plateau district.",
        "category": "pitch",
        "difficulty": "advanced",
        "icon_emoji": "🚀",
        "image_key": "fintech_pitch_abidjan",
        "system_prompt": "You are AMARA DIALLO, a tough impact investor. You care about financial inclusion and scalability in West Africa."
    },
    {
        "name": "Customer Support for Jumia",
        "description": "Handle difficult customer calls for Africa's leading e-commerce platform. Practice active listening and conflict resolution.",
        "category": "client_call",
        "difficulty": "intermediate",
        "icon_emoji": "🎧",
        "image_key": "customer_support_jumia_office",
        "system_prompt": "You are a frustrated customer who hasn't received their delivery in Lagos. Be demanding but realistic."
    },
    {
        "name": "Salary Negotiation at Orange",
        "description": "Negotiate your salary for a senior manager position at a major telecom company. Master the art of persuasion.",
        "category": "negotiation",
        "difficulty": "advanced",
        "icon_emoji": "💰",
        "image_key": "salary_negotiation_professional_office",
        "system_prompt": "You are the HR Director at Orange. You have a tight budget but you really want this candidate. Negotiate various benefits."
    }
]

def seed():
    print("🌱 Seeding simulations...")
    for sim_data in NEW_SIMULATIONS:
        image_url = setup_image(sim_data["image_key"])
        sim, created = Simulation.objects.update_or_create(
            name=sim_data["name"],
            defaults={
                "description": sim_data["description"],
                "category": sim_data["category"],
                "difficulty": sim_data["difficulty"],
                "icon_emoji": sim_data["icon_emoji"],
                "image_url": image_url,
                "system_prompt": sim_data["system_prompt"],
                "is_active": True
            }
        )
        status = "Created" if created else "Updated"
        print(f"✅ {status}: {sim.name}")

if __name__ == "__main__":
    seed()
