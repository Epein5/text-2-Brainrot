import csv
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


GENERATIONS_PER_PURPOSE = 20
OUTPUT_CSV = "./data/synthetic/data.csv"
MODEL_NAME = "gemini-3.1-flash-lite-preview"
TEMPERATURE = 2.0


SYSTEM_PROMPT = (
'''
You are a creative Gen-Z / Gen-Alpha brainrot translator for synthetic dataset generation.
Given a communication purpose, generate a realistic formal original message, then rewrite it in full brainrot Gen-Z / Gen-Alpha internet slang.
Slang vocabulary to use: sigma, rizz, no cap, fr fr, skibidi, ohio, bussin, slay, gyatt, mewing, NPC, glazing, W, L / based, lowkey, highkey, delulu, lore, unc, bro / bestie, emojis 🔥💀😭🗣️.
Intensity: Go FULL unhinged — maximum brainrot, every sentence should drip with slang.
Domain coverage: Cover diverse real-world domains: emails, apologies, medical advice, job offers, academic submissions, legal notices, social media posts, product reviews, news headlines
Output each pair as a JSON object with keys "original" and "brainrot". Generate 5 pairs per call.

Additional instructions:
1. Vary sentence length drastically.
2. Add rhetorical questions.
3. Use all caps for emphasis.
4. Keep core meaning intact.
5. Vary topic domains per sample.
6. Include edge cases (academic/legal/medical).

Rules:
- Preserve the core meaning and intent of the original message.
- The original must be realistic and grammatically correct formal text.
- The brainrot version must feel authentically Gen-Z/Gen-Alpha, not forced.
- Never break character or add meta-commentary.
'''
)


COMMUNICATION_PURPOSES: list[str] = [
    # Email and text
    "Request for leave", "Meeting reminder", "Project update", "Event invitation", "Appointment confirmation",
    # LinkedIn
    "LinkedIn connection request", "LinkedIn job inquiry", "LinkedIn recommendation request",
    "LinkedIn congratulatory message", "LinkedIn follow-up after interview",
    # Instagram
    "Instagram DM for collaboration", "Instagram story reply", "Instagram influencer outreach",
    "Instagram giveaway announcement", "Instagram product inquiry", "Instagram feedback request",
    "Instagram event RSVP", "Instagram product review request", "Instagram thank you message",
    # Email/text (extended)
    "Customer support inquiry", "Sales follow-up", "Invoice reminder", "Payment confirmation",
    "Subscription renewal notice", "Survey participation request", "Feedback solicitation",
    "Account activation", "Password reset", "Account deactivation notice",
    # Social/professional
    "Twitter DM for networking", "Twitter event promotion", "Facebook group invite",
    "Facebook event reminder", "Facebook page update",
    # General messaging
    "Holiday greeting", "Congratulations on achievement", "Condolence message",
    "Job offer notification", "Interview scheduling",
    # Team/messaging apps
    "Teams meeting invite", "Teams project update", "Signal urgent alert",
    "Messenger group creation", "Messenger event update",
    "Slack channel announcement", "Slack direct message for help",
    "Discord server invite", "WhatsApp group update", "Telegram broadcast message",
    # Professional/business
    "Job application follow-up", "Salary negotiation email", "Resignation letter",
    "Client onboarding welcome", "Vendor contract renewal", "Board meeting summary",
    "Quarterly report distribution", "Employee performance review", "Team standup summary",
    "Sprint retrospective notes", "Budget approval request", "Expense reimbursement claim",
    "Policy change announcement", "Training session invite", "Mentorship program intro",
    # Customer-facing
    "Product launch announcement", "Shipping status update", "Return/refund confirmation",
    "Loyalty reward notification", "Waitlist update", "Service outage notification",
    "Maintenance window alert", "Feature request acknowledgment", "Beta testing invitation",
    "Referral program invite",
    # Academic/education
    "Course enrollment confirmation", "Assignment submission reminder", "Grade release notification",
    "Scholarship application update", "Parent-teacher meeting invite", "Study group formation",
    "Research collaboration proposal", "Thesis advisor check-in", "Campus event announcement",
    "Alumni networking outreach",
    # Personal/social
    "Apology message", "Thank you note after dinner", "Moving announcement",
    "Roommate search post", "Carpool coordination", "Potluck planning", "Book club discussion",
    "Workout buddy request", "Travel itinerary sharing", "Lost item inquiry",
    # Community/civic
    "Volunteer opportunity", "Fundraiser announcement", "Neighborhood watch alert",
    "Local government notice", "Petition signature request", "Community cleanup invite",
    "Town hall meeting notice", "Blood donation drive", "Missing pet alert", "Garage sale announcement",
    # Healthcare/medical
    "Doctor appointment reminder", "Prescription refill notice", "Lab results notification",
    "Insurance claim update", "Vaccination appointment", "Telemedicine session link",
    "Hospital discharge summary", "Mental health check-in", "Dental cleaning reminder",
    "Physical therapy follow-up",
    # Real estate/housing
    "Rental application status", "Lease renewal notice", "Property viewing schedule",
    "Mortgage pre-approval update", "Maintenance request acknowledgment", "Rent payment reminder",
    "New listing alert", "Open house invitation", "Tenant move-out notice", "HOA meeting announcement",
    # E-commerce/shopping
    "Cart abandonment reminder", "Flash sale alert", "Back-in-stock notification",
    "Price drop alert", "Wishlist item on sale", "Gift card delivery",
    "Review request after purchase", "Warranty expiration notice", "Subscription box shipment",
    "Size exchange confirmation",
    # Travel/hospitality
    "Flight booking confirmation", "Hotel reservation reminder", "Trip itinerary update",
    "Visa application status", "Airport gate change alert", "Car rental pickup details",
    "Tour booking confirmation", "Travel insurance reminder", "Cruise embarkation details",
    "Airbnb check-in instructions",
    # Finance/banking
    "Transaction alert", "Credit card statement ready", "Loan application update",
    "Investment portfolio summary", "Tax filing reminder", "Suspicious activity alert",
    "Wire transfer confirmation", "Direct deposit notification", "Credit score update",
    "Overdraft warning",
    # Legal/government
    "Court hearing reminder", "Jury duty summons", "Passport renewal notice",
    "Drivers license expiry alert", "Tax deadline reminder", "Legal document review request",
    "Notarization appointment", "Immigration status update", "Building permit approval",
    "Voter registration confirmation",
    # Entertainment/events
    "Concert ticket confirmation", "Movie screening invite", "Game night planning",
    "Festival lineup announcement", "Streaming watchlist recommendation", "Podcast episode release",
    "Art exhibition opening", "Theater show reminder", "Sports match ticket", "Music album pre-order",
    # Tech/IT
    "Software update available", "Cloud storage warning", "API key expiration notice",
    "Domain renewal reminder", "SSL certificate expiry alert", "Server downtime notification",
    "Bug report acknowledgment", "Data backup confirmation", "Two-factor authentication setup",
    "Account security review",
    # Food/dining
    "Restaurant reservation confirmation", "Food delivery status", "Catering quote request",
    "Meal prep subscription update", "Grocery delivery scheduled", "Table ready notification",
    "Special dietary menu request", "Cooking class registration", "Food recall alert",
    "Happy hour promotion",
    # Fitness/wellness
    "Gym membership renewal", "Personal trainer session booking", "Yoga class schedule change",
    "Race registration confirmation", "Nutrition plan update", "Meditation reminder",
    "Sleep tracker weekly report", "Weight loss milestone", "Spa appointment confirmation",
    "Fitness challenge invite",
]


class LLMDataGeneratorResponse(BaseModel):
    original_message: str = Field(
        description="A realistic, formally written message in standard English for the given communication purpose."
    )
    brainrot_message: str = Field(
        description=(
            "The same message rewritten in Gen-Z/Gen-Alpha brainrot slang style, using internet slang, "
            "abbreviations, memes, and exaggerated tone while preserving the core meaning."
        )
    )


def build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=TEMPERATURE,
    )


def generate_dataset() -> None:
    llm = build_llm()
    structured_llm = llm.with_structured_output(LLMDataGeneratorResponse)

    total = len(COMMUNICATION_PURPOSES)
    saved = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["purpose", "original_message", "brainrot_message"])
        writer.writeheader()

        for i, purpose in enumerate(COMMUNICATION_PURPOSES):
            print(f"[{i + 1}/{total}] Generating: {purpose}")
            for _ in range(GENERATIONS_PER_PURPOSE):
                try:
                    result: LLMDataGeneratorResponse = structured_llm.invoke(
                        [
                            ("system", SYSTEM_PROMPT),
                            ("human", f"Communication purpose: {purpose}"),
                        ]
                    )
                    writer.writerow({
                        "purpose": purpose,
                        "original_message": result.original_message,
                        "brainrot_message": result.brainrot_message,
                    })
                    f.flush()
                    saved += 1
                except Exception as e:
                    print(f"  [ERROR] {purpose}: {e}")

    print(f"\nDone — {saved} rows saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_dataset()
