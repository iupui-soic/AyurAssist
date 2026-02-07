'use client'

import { useState } from 'react'

// ── Types matching Python API response ──────────────────
interface OttamooliRemedy {
  medicine_name: string
  sanskrit_name: string
  part_used: string
  preparation: string
  dosage: string
  timing: string
  duration: string
}

interface ClassicalFormulation {
  name: string
  english_name: string
  form: string
  dosage: string
  reference_text: string
}

interface PanchakarmaTherapy {
  therapy_name: string
  description: string
  indication: string
}

interface DietaryAdvice {
  foods_to_favor: string[]
  foods_to_avoid: string[]
  specific_dietary_rules: string
}

interface TreatmentInfo {
  condition_name: string
  sanskrit_name: string
  brief_description: string
  dosha_involvement: string
  nidana_causes: string[]
  purvarupa_prodromal_symptoms: string[]
  rupa_symptoms: string[]
  ottamooli_single_remedies: OttamooliRemedy[]
  classical_formulations: ClassicalFormulation[]
  panchakarma_treatments: PanchakarmaTherapy[]
  pathya_dietary_advice: DietaryAdvice
  vihara_lifestyle: string[]
  yoga_exercises: string[]
  modern_correlation: string
  prognosis: string
  warning_signs: string[]
  disclaimer: string
  note?: string
  snomed_code?: string
  snomed_name?: string
}

interface ConditionResult {
  input_entity: string
  match_type: string
  match_score: number
  ita_id: string
  ayurveda_term: string
  sanskrit: string
  snomed_code: string
  snomed_name: string
  who_description: string
  treatment_info: TreatmentInfo
}

interface APIResponse {
  input_text: string
  entities_extracted: { text: string; label: string; score: number }[]
  conditions_matched: number
  results: ConditionResult[]
}

// ✅ Corrected: Defaults to your deployed Modal endpoint
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://aravindkv28--ayurparam-service-ayurengine-process-query.modal.run'

export default function Home() {
  const [inputText, setInputText] = useState('')
  const [results, setResults] = useState<ConditionResult[]>([])
  const [entities, setEntities] = useState<{ text: string; label: string; score: number }[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const [hasSearched, setHasSearched] = useState(false)

  const handleAnalyze = async () => {
    if (!inputText.trim()) return
    setIsAnalyzing(true)
    setError('')
    setHasSearched(true)

    try {
      // ✅ FIX: Using backticks for template literal and calling the root of the Modal endpoint
      const res = await fetch(`${API_BASE}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      })

      if (!res.ok) throw new Error(`API error: ${res.status}`)
      
      const data: APIResponse = await res.json()
      
      // ✅ Handle cases where results might be empty
      if (data.results && data.results.length > 0) {
        setResults(data.results)
        setEntities(data.entities_extracted || [])
      } else {
        setResults([])
        setError('No matching Ayurvedic conditions found in the database or via AI.')
      }
    } catch (err: any) {
      console.error("Connection Error:", err)
      setError('Could not connect to the cloud AI. Check your internet or Modal deployment.')
      setResults([])
      setEntities([])
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') { e.preventDefault(); handleAnalyze() }
  }

  const hasLLMData = (info: TreatmentInfo) =>
    info && info.ottamooli_single_remedies && info.ottamooli_single_remedies.length > 0

  return (
    <div style={{ position: 'relative', zIndex: 1 }}>
      <div style={{ maxWidth: 900, margin: '0 auto', padding: '24px 20px 60px' }}>

        {/* ─── HEADER ─── */}
        <header style={{ textAlign: 'center', padding: '48px 0 36px' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: 'var(--green-light)', color: 'var(--green-deep)',
            fontSize: 11, fontWeight: 600, letterSpacing: 1.5,
            textTransform: 'uppercase' as const, padding: '6px 16px', borderRadius: 100, marginBottom: 16
          }}>
            🌿 WHO ITA Standards · SNOMED CT Mapped
          </div>
          <h1 style={{
            fontSize: 44, fontWeight: 700,
            color: 'var(--green-deep)', letterSpacing: -1, lineHeight: 1.1
          }}>
            Ayur<span style={{ color: 'var(--green-soft)', fontWeight: 300, fontStyle: 'italic' }}>Assist</span>
          </h1>
          <p style={{ fontSize: 15, color: 'var(--text-light)', marginTop: 8 }}>
            Ayurveda–SNOMED Clinical Decision Support
          </p>
          <div style={{
            width: 60, height: 3, margin: '20px auto 0', borderRadius: 2,
            background: 'linear-gradient(90deg, var(--green-soft), var(--amber))'
          }} />
        </header>

        {/* ─── SEARCH ─── */}
        <section style={{
          background: 'var(--bg-card)', borderRadius: 28, padding: 32,
          boxShadow: '0 4px 16px rgba(0,0,0,0.08)', border: '1px solid var(--border)', marginBottom: 28
        }}>
          <p style={{ fontSize: 20, fontWeight: 500, marginBottom: 4 }}>
            Describe your symptoms
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-light)', marginBottom: 16 }}>
            Enter symptoms in plain language — our engine will identify medical entities
          </p>
          <div style={{ display: 'flex', gap: 10, alignItems: 'stretch' }}>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="e.g., headache, stomach pain, cough with fever..."
              style={{
                flex: 1, padding: '14px 20px', fontSize: 16,
                border: '2px solid var(--border)', borderRadius: 12,
                background: 'var(--bg-cream)', color: 'var(--text-dark)', outline: 'none'
              }}
            />
            <button
              onClick={handleAnalyze}
              disabled={!inputText.trim() || isAnalyzing}
              style={{
                padding: '14px 28px',
                background: 'linear-gradient(135deg, var(--green-deep), var(--green-mid))',
                color: '#fff', border: 'none', borderRadius: 12, fontSize: 15, fontWeight: 600,
                cursor: 'pointer', opacity: (!inputText.trim() || isAnalyzing) ? 0.5 : 1
              }}
            >
              {isAnalyzing ? "Analyzing..." : "Analyze"}
            </button>
          </div>
        </section>

        {/* ─── ERROR ─── */}
        {error && (
          <div style={{
            background: 'var(--red-soft)', border: '1px solid #F5C6CB', borderRadius: 12,
            padding: '14px 20px', marginBottom: 20, color: 'var(--red-warn)', fontSize: 14
          }}>
            ⚠️ {error}
          </div>
        )}

        {/* ─── RESULTS ─── */}
        {results.map((result, index) => {
          const info = result.treatment_info;
          const hasLLM = hasLLMData(info);

          return (
            <div key={index} style={{ marginBottom: 32 }}>
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '14px 20px', background: 'var(--green-deep)', borderRadius: 12, marginBottom: 20, color: '#fff'
              }}>
                <div>
                    <div style={{ fontSize: 16, fontWeight: 600 }}>Ayurvedic Match Found</div>
                    <div style={{ fontSize: 12, opacity: 0.8 }}>Matched: {result.ayurveda_term}</div>
                </div>
                <div style={{ fontSize: 28, fontWeight: 700 }}>{result.match_score}%</div>
              </div>

              <Card>
                <h2 style={{ fontSize: 30, color: 'var(--green-deep)' }}>{result.ayurveda_term}</h2>
                <p style={{ color: 'var(--terra)', fontStyle: 'italic' }}>{result.sanskrit || info.sanskrit_name}</p>
                <div style={{ fontSize: 13, color: 'var(--text-light)', marginTop: 10 }}>
                  SNOMED CT: <span style={{ background: 'var(--bg-warm)', padding: '2px 6px' }}>{result.snomed_code}</span> {result.snomed_name}
                </div>
              </Card>

              <SectionCard icon="📋" iconBg="var(--green-light)" title="Description">
                <p style={{ fontSize: 15, lineHeight: 1.6, color: 'var(--text-mid)' }}>{info.brief_description || result.who_description}</p>
              </SectionCard>

              {hasLLM && (
                <>
                  <SectionCard icon="🌿" iconBg="var(--green-light)" title="Ottamooli Remedies">
                    {info.ottamooli_single_remedies.map((r, i) => (
                      <div key={i} style={{ padding: 10, borderBottom: '1px solid var(--border)' }}>
                        <strong>{r.medicine_name}</strong> - {r.preparation} ({r.dosage})
                      </div>
                    ))}
                  </SectionCard>

                  <SectionCard icon="🍽️" iconBg="var(--green-light)" title="Diet (Pathya)">
                    <div style={{ color: 'var(--green-deep)' }}>Favor: {info.pathya_dietary_advice.foods_to_favor.join(', ')}</div>
                    <div style={{ color: 'var(--red-warn)', marginTop: 5 }}>Avoid: {info.pathya_dietary_advice.foods_to_avoid.join(', ')}</div>
                  </SectionCard>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  )
}

/* ─── Reusable UI Components ─── */
function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{
      background: 'var(--bg-card)', borderRadius: 20, padding: 24,
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)', border: '1px solid var(--border)',
      marginBottom: 12, ...style
    }}>{children}</div>
  )
}

function SectionCard({ icon, iconBg, title, children }: { icon: string; iconBg: string; title: string; children: React.ReactNode }) {
  return (
    <Card>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
        <div style={{ width: 36, height: 36, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', background: iconBg }}>{icon}</div>
        <span style={{ fontSize: 18, fontWeight: 600 }}>{title}</span>
      </div>
      {children}
    </Card>
  )
}
