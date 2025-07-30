class IllustrationGenerator:
    def __init__(self):
        self.base_model = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        )
        self.character_embeddings = self.load_character_embeddings()
        
    def generate_illustration(self, 
                            scene_description: str,
                            character_profile: CharacterProfile,
                            style: str = "children_book") -> Image:
        
        # Create consistent character prompt
        character_prompt = self.build_character_prompt(character_profile)
        
        # Combine scene + character + style
        full_prompt = f"""
        {scene_description}, featuring {character_prompt},
        in {style} illustration style, vibrant colors,
        child-friendly, warm lighting, detailed background
        """
        
        # Generate with consistency pipeline
        image = self.base_model(
            prompt=full_prompt,
            negative_prompt=self.get_safety_negative_prompt(),
            guidance_scale=7.5,
            num_inference_steps=30
        ).images[0]
        
        # Post-process for consistency
        return self.ensure_character_consistency(image, character_profile)
