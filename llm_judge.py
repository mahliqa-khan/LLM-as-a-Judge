"""
LLM-as-a-Judge: A simple framework for using LLMs to evaluate text.
"""

import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import os


@dataclass
class JudgmentResult:
    """Result of an LLM judgment."""
    score: float
    explanation: str
    criteria: str
    raw_response: str


class LLMJudge:
    """
    A simple LLM-as-a-judge implementation that can evaluate text based on custom criteria.

    Supports OpenAI and Anthropic APIs.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM Judge.

        Args:
            provider: API provider ("openai" or "anthropic")
            model: Model name (defaults to gpt-4 for OpenAI, claude-3-5-sonnet for Anthropic)
            api_key: API key (if not provided, reads from environment)
        """
        self.provider = provider.lower()

        if self.provider == "openai":
            import openai
            self.model = model or "gpt-4o"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic
            self.model = model or "claude-3-5-sonnet-20241022"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'")

    def judge(
        self,
        text: str,
        criteria: str,
        scale: str = "1-10",
        context: Optional[str] = None,
        reference: Optional[str] = None
    ) -> JudgmentResult:
        """
        Evaluate text based on given criteria.

        Args:
            text: The text to evaluate
            criteria: Evaluation criteria (e.g., "helpfulness", "accuracy", "coherence")
            scale: Scoring scale (default "1-10")
            context: Optional context for evaluation
            reference: Optional reference text for comparison

        Returns:
            JudgmentResult with score and explanation
        """
        prompt = self._build_prompt(text, criteria, scale, context, reference)

        if self.provider == "openai":
            response = self._call_openai(prompt)
        else:
            response = self._call_anthropic(prompt)

        return self._parse_response(response, criteria)

    def compare(
        self,
        text_a: str,
        text_b: str,
        criteria: str,
        context: Optional[str] = None
    ) -> Dict[str, Union[str, JudgmentResult]]:
        """
        Compare two texts and determine which is better.

        Args:
            text_a: First text to compare
            text_b: Second text to compare
            criteria: Evaluation criteria
            context: Optional context for comparison

        Returns:
            Dictionary with winner and individual judgments
        """
        prompt = self._build_comparison_prompt(text_a, text_b, criteria, context)

        if self.provider == "openai":
            response = self._call_openai(prompt)
        else:
            response = self._call_anthropic(prompt)

        return self._parse_comparison(response, criteria)

    def batch_judge(
        self,
        texts: List[str],
        criteria: str,
        scale: str = "1-10",
        context: Optional[str] = None
    ) -> List[JudgmentResult]:
        """
        Evaluate multiple texts with the same criteria.

        Args:
            texts: List of texts to evaluate
            criteria: Evaluation criteria
            scale: Scoring scale
            context: Optional context for evaluation

        Returns:
            List of JudgmentResults
        """
        return [
            self.judge(text, criteria, scale, context)
            for text in texts
        ]

    def _build_prompt(
        self,
        text: str,
        criteria: str,
        scale: str,
        context: Optional[str],
        reference: Optional[str]
    ) -> str:
        """Build the evaluation prompt."""
        prompt = f"""You are an expert evaluator. Please evaluate the following text based on the specified criteria.

Criteria: {criteria}
Scale: {scale}
"""

        if context:
            prompt += f"\nContext: {context}"

        if reference:
            prompt += f"\nReference: {reference}"

        prompt += f"""

Text to evaluate:
{text}

Please provide your evaluation in the following JSON format:
{{
    "score": <numerical score on the specified scale>,
    "explanation": "<detailed explanation of your judgment>"
}}
"""
        return prompt

    def _build_comparison_prompt(
        self,
        text_a: str,
        text_b: str,
        criteria: str,
        context: Optional[str]
    ) -> str:
        """Build the comparison prompt."""
        prompt = f"""You are an expert evaluator. Please compare the following two texts based on the specified criteria.

Criteria: {criteria}
"""

        if context:
            prompt += f"\nContext: {context}"

        prompt += f"""

Text A:
{text_a}

Text B:
{text_b}

Please provide your comparison in the following JSON format:
{{
    "winner": "<'A', 'B', or 'tie'>",
    "score_a": <score for text A (1-10)>,
    "score_b": <score for text B (1-10)>,
    "explanation": "<detailed explanation of your judgment>"
}}
"""
        return prompt

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a fair and thorough evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.content[0].text

    def _parse_response(self, response: str, criteria: str) -> JudgmentResult:
        """Parse the LLM response into a JudgmentResult."""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            return JudgmentResult(
                score=float(data['score']),
                explanation=data['explanation'],
                criteria=criteria,
                raw_response=response
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: create a result with the raw response
            return JudgmentResult(
                score=0.0,
                explanation=f"Failed to parse response: {str(e)}",
                criteria=criteria,
                raw_response=response
            )

    def _parse_comparison(self, response: str, criteria: str) -> Dict:
        """Parse the comparison response."""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            return {
                'winner': data['winner'],
                'score_a': float(data['score_a']),
                'score_b': float(data['score_b']),
                'explanation': data['explanation'],
                'criteria': criteria,
                'raw_response': response
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return {
                'winner': 'error',
                'score_a': 0.0,
                'score_b': 0.0,
                'explanation': f"Failed to parse response: {str(e)}",
                'criteria': criteria,
                'raw_response': response
            }
