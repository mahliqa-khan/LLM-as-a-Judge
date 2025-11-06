"""
PDF Verification Module - Verify extracted information against source PDFs.
"""

import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from llm_judge import LLMJudge


@dataclass
class VerificationResult:
    """Result of verifying an extracted claim against the PDF."""
    claim: str
    is_verified: bool
    confidence: float  # 0-10 scale
    explanation: str
    relevant_sections: List[str]
    page_numbers: List[int]


class PDFVerifier:
    """
    Verifies extracted information against source PDFs using LLM-as-a-Judge.

    Uses chunking and retrieval to handle large PDFs efficiently.
    """

    def __init__(self, provider: str = "openai", model: str = None):
        """
        Initialize the PDF Verifier.

        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Specific model to use (defaults to gpt-4o-mini for cheapest cost)
        """
        # Use cheapest models by default
        if model is None:
            if provider == "openai":
                model = "gpt-4o-mini"
            elif provider == "anthropic":
                model = "claude-3-5-haiku-20241022"

        self.judge = LLMJudge(provider=provider, model=model)
        self.pdf_chunks = []
        self.pdf_text = ""

    def load_pdf(self, pdf_path: str):
        """
        Load and chunk a PDF file.

        Args:
            pdf_path: Path to the PDF file
        """
        try:
            import pymupdf  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install it with: pip install pymupdf"
            )

        doc = pymupdf.open(pdf_path)
        self.pdf_chunks = []

        # Extract text page by page
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                self.pdf_chunks.append({
                    'text': text,
                    'page': page_num
                })

        doc.close()
        self.pdf_text = "\n\n".join([chunk['text'] for chunk in self.pdf_chunks])
        print(f"Loaded PDF: {len(self.pdf_chunks)} pages")

    def _find_relevant_chunks(self, claim: str, top_k: int = 3) -> List[Dict]:
        """
        Find the most relevant chunks for a given claim.

        Uses simple keyword matching. For better results, consider using embeddings.

        Args:
            claim: The claim to verify
            top_k: Number of top chunks to return

        Returns:
            List of relevant chunks with scores
        """
        # Extract key terms from the claim (simple approach)
        claim_lower = claim.lower()
        words = re.findall(r'\b\w+\b', claim_lower)
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        # Score each chunk based on keyword matches
        scored_chunks = []
        for chunk in self.pdf_chunks:
            chunk_lower = chunk['text'].lower()
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            if score > 0:
                scored_chunks.append({
                    'chunk': chunk,
                    'score': score
                })

        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return [item['chunk'] for item in scored_chunks[:top_k]]

    def verify_claim(self, claim: str, context: str = None) -> VerificationResult:
        """
        Verify a single extracted claim against the PDF.

        Args:
            claim: The claim to verify
            context: Optional context about what this claim represents

        Returns:
            VerificationResult with verification details
        """
        if not self.pdf_chunks:
            raise ValueError("No PDF loaded. Call load_pdf() first.")

        # Find relevant sections
        relevant_chunks = self._find_relevant_chunks(claim, top_k=3)

        if not relevant_chunks:
            # No relevant sections found - likely false
            return VerificationResult(
                claim=claim,
                is_verified=False,
                confidence=0.0,
                explanation="No relevant sections found in the PDF that could support this claim.",
                relevant_sections=[],
                page_numbers=[]
            )

        # Build verification prompt
        sections_text = "\n\n---\n\n".join([
            f"[Page {chunk['page']}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ])

        verification_prompt = f"""You are verifying whether an extracted claim is supported by the source document.

EXTRACTED CLAIM:
{claim}

RELEVANT SECTIONS FROM SOURCE PDF:
{sections_text}

Please verify if the claim is accurately supported by these sections. Consider:
1. Is the claim factually present in the source?
2. Is the claim accurately represented (not distorted or exaggerated)?
3. Is there sufficient evidence in these sections?

Respond in JSON format:
{{
    "is_verified": true or false,
    "confidence": <score 0-10, where 10 is absolutely certain>,
    "explanation": "<detailed explanation of your verification>"
}}"""

        # Call the judge
        if self.judge.provider == "openai":
            response = self.judge._call_openai(verification_prompt)
        else:
            response = self.judge._call_anthropic(verification_prompt)

        # Parse response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            return VerificationResult(
                claim=claim,
                is_verified=data['is_verified'],
                confidence=float(data['confidence']),
                explanation=data['explanation'],
                relevant_sections=[chunk['text'] for chunk in relevant_chunks],
                page_numbers=[chunk['page'] for chunk in relevant_chunks]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return VerificationResult(
                claim=claim,
                is_verified=False,
                confidence=0.0,
                explanation=f"Error parsing verification response: {str(e)}",
                relevant_sections=[chunk['text'] for chunk in relevant_chunks],
                page_numbers=[chunk['page'] for chunk in relevant_chunks]
            )

    def verify_extraction(self, extracted_data: str) -> List[VerificationResult]:
        """
        Verify multiple claims from extracted data.

        Args:
            extracted_data: The extracted data (can be text, JSON, or key-value pairs)

        Returns:
            List of VerificationResults
        """
        # Parse the extracted data into individual claims
        claims = self._parse_extraction(extracted_data)

        results = []
        for claim in claims:
            result = self.verify_claim(claim)
            results.append(result)

        return results

    def _parse_extraction(self, extracted_data: str) -> List[str]:
        """
        Parse extracted data into individual claims to verify.

        Handles various formats:
        - Line-by-line claims
        - JSON with key-value pairs
        - Bullet points
        """
        claims = []

        # Try parsing as JSON first
        try:
            data = json.loads(extracted_data)
            if isinstance(data, dict):
                # Convert key-value pairs to claims
                for key, value in data.items():
                    claims.append(f"{key}: {value}")
            elif isinstance(data, list):
                claims = [str(item) for item in data]
            return claims
        except json.JSONDecodeError:
            pass

        # Parse as plain text - split by lines/bullets
        lines = extracted_data.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Remove common bullet point characters
            line = re.sub(r'^[-*•]\s*', '', line)
            if line:
                claims.append(line)

        return claims

    def print_verification_report(self, results: List[VerificationResult]):
        """
        Print a formatted verification report.

        Args:
            results: List of VerificationResults
        """
        print("\n" + "="*80)
        print("VERIFICATION REPORT")
        print("="*80 + "\n")

        verified_count = sum(1 for r in results if r.is_verified)
        total_count = len(results)

        print(f"Total Claims: {total_count}")
        print(f"Verified: {verified_count}")
        print(f"Not Verified: {total_count - verified_count}")
        print(f"Success Rate: {(verified_count/total_count*100):.1f}%\n")

        print("="*80)
        print("DETAILED RESULTS")
        print("="*80 + "\n")

        for i, result in enumerate(results, 1):
            status = "✓ VERIFIED" if result.is_verified else "✗ NOT VERIFIED"
            print(f"{i}. {status} (Confidence: {result.confidence:.1f}/10)")
            print(f"   Claim: {result.claim}")
            print(f"   Explanation: {result.explanation}")
            if result.page_numbers:
                print(f"   Relevant Pages: {', '.join(map(str, result.page_numbers))}")
            print()
