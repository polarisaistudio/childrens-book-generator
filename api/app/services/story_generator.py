class StoryGenerator:
    def __init__(self):
        self.base_model = "gpt-4-turbo"
        self.fine_tuned_model = self.load_custom_model()
        
    def generate_story(self, user_profile: UserProfile) -> Story:
        prompt = self.build_personalized_prompt(user_profile)
        
        # Multi-step generation for better quality
        outline = self.generate_outline(prompt)
        chapters = self.generate_chapters(outline, user_profile)
        
        return Story(
            outline=outline,
            chapters=chapters,
            age_appropriate_score=self.safety_score(chapters),
            reading_level=self.analyze_reading_level(chapters)
        )
        
    def build_personalized_prompt(self, profile: UserProfile) -> str:
        return f"""
        Create a children's story for {profile.child_name}, age {profile.age}.
        Interests: {', '.join(profile.interests)}
        Character traits: {profile.personality}
        Preferred themes: {profile.themes}
        
        Story requirements:
        - Age-appropriate for {profile.age} years old
        - Include {profile.child_name} as main character
        - 8-12 pages long
        - Positive moral lesson
        - Adventure/discovery theme
        """
