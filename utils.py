from typing import List
from enum import Enum

from pydantic import BaseModel

class LawDomain(Enum):
    CRIMINAL = 0
    CIVIL = 1
    PUBLIC = 2
    CONSTITUTIONAL = 3
    REVISION = 4

class Article(BaseModel):
    source: str
    title: str
    content: str

class Case(BaseModel):
    description: str
    related_articles: List[Article]
    outcome: bool
    domain: LawDomain
