import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AyurAssist - Ayurveda SNOMED Knowledge Portal',
  description: 'Connect modern symptoms to traditional Ayurvedic terminology',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}