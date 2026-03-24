from __future__ import annotations

from eda_preprocessing.analyzers.answer_length import AnswerLengthAnalyzer
from eda_preprocessing.analyzers.answer_source import AnswerSourceAnalyzer
from eda_preprocessing.analyzers.answer_type import AnswerTypeAnalyzer
from eda_preprocessing.analyzers.cross_image_dependency import CrossImageDependencyAnalyzer
from eda_preprocessing.analyzers.diacritics import DiacriticsAnalyzer
from eda_preprocessing.analyzers.image_resolution import ImageResolutionAnalyzer
from eda_preprocessing.analyzers.multi_image_count import MultiImageCountAnalyzer
from eda_preprocessing.analyzers.question_type import QuestionTypeAnalyzer
from eda_preprocessing.analyzers.text_density import TextDensityAnalyzer
from eda_preprocessing.analyzers.tokenization_ambiguity import TokenizationAmbiguityAnalyzer


def build_analyzers() -> list[object]:
    return [
        AnswerLengthAnalyzer(),
        AnswerTypeAnalyzer(),
        QuestionTypeAnalyzer(),
        AnswerSourceAnalyzer(),
        ImageResolutionAnalyzer(),
        TextDensityAnalyzer(),
        MultiImageCountAnalyzer(),
        CrossImageDependencyAnalyzer(),
        DiacriticsAnalyzer(),
        TokenizationAmbiguityAnalyzer(),
    ]
