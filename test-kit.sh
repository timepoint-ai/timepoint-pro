# Set up for real LLM calls
export OPENROUTER_API_KEY="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"

# Clean slate
rm -f timepoint.db
rm -rf reports/

# Test 1: Historical training with founding fathers
echo "=== TEST 1: Historical Context Training ==="
poetry run python cli.py mode=train training.context=founding_fathers_1789

# Test 2: Verify database persistence
echo -e "\n=== TEST 2: Database Verification ==="
poetry run python -c "
from storage import GraphStore
from sqlmodel import Session, select
from schemas import Entity
import json

store = GraphStore()
with Session(store.engine) as session:
    entities = session.exec(select(Entity)).all()
    print(f'Total entities: {len(entities)}')
    
    for entity in entities:
        print(f'\n{entity.entity_id}:')
        print(f'  Role: {entity.entity_metadata.get(\"role\")}')
        print(f'  Age: {entity.entity_metadata.get(\"age\")}')
        print(f'  Location: {entity.entity_metadata.get(\"location\")}')
        print(f'  Knowledge items: {len(entity.entity_metadata.get(\"knowledge_state\", []))}')
        print(f'  First 2 knowledge: {entity.entity_metadata.get(\"knowledge_state\", [])[:2]}')
        print(f'  Temporal awareness: {entity.entity_metadata.get(\"temporal_awareness\", \"\")[:80]}...')
"

# Test 3: Evaluate historical entities
echo -e "\n=== TEST 3: Historical Entity Evaluation ==="
poetry run python cli.py mode=evaluate

# Test 4: Renaissance context training
echo -e "\n=== TEST 4: Second Historical Context ==="
poetry run python cli.py mode=train training.context=renaissance_florence_1504

# Test 5: Mixed evaluation
echo -e "\n=== TEST 5: Mixed Context Evaluation ==="
poetry run python cli.py mode=evaluate

# Test 6: Autopilot with multiple sizes
echo -e "\n=== TEST 6: Autopilot Stress Test ==="
poetry run python cli.py mode=autopilot autopilot.graph_sizes="[5,10,15]"

# Test 7: Check reports generated
echo -e "\n=== TEST 7: Report Generation ==="
ls -lh reports/
echo -e "\nReport contents:"
cat reports/train_report_*.md | head -30

# Test 8: Final database state
echo -e "\n=== TEST 8: Final Database State ==="
poetry run python -c "
from storage import GraphStore
from sqlmodel import Session, select
from schemas import Entity

store = GraphStore()
with Session(store.engine) as session:
    entities = session.exec(select(Entity)).all()
    
    by_type = {}
    for e in entities:
        ctx = e.entity_metadata.get('historical_context', 'generic')
        by_type[ctx] = by_type.get(ctx, 0) + 1
    
    print(f'Total entities: {len(entities)}')
    print(f'By context:')
    for ctx, count in by_type.items():
        print(f'  {ctx}: {count}')
"

echo -e "\n=== ALL TESTS COMPLETE ==="