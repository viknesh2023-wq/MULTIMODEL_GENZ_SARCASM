import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Footer } from './components/Footer';

// Pages
import { Overview } from './pages/Overview';
import { SarcasmDetection } from './pages/SarcasmDetection';
import { TextAnalysis } from './pages/TextAnalysis';
import { MultimodalAnalysis } from './pages/MultimodalAnalysis';
import { ModelArchitecture } from './pages/ModelArchitecture';
import { ModelPerformance } from './pages/ModelPerformance';
import { FeatureExtraction } from './pages/FeatureExtraction';
import { Dataset } from './pages/Dataset';
import { ResearchNovelty } from './pages/ResearchNovelty';
import { AboutProject } from './pages/AboutProject';

import { getHealth, getModelStatus } from './services/api';
import type { NavPage } from './types';

export default function App() {
  const [currentPage, setCurrentPage] = useState<NavPage>('overview');
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [multimodalReady, setMultimodalReady] = useState<boolean>(false);

  // Poll backend health & model status
  useEffect(() => {
    let mounted = true;

    const checkStatus = async () => {
      try {
        const health = await getHealth();
        if (!mounted) return;
        setBackendOnline(health.status === 'ok');
        setMultimodalReady(Boolean(health.multimodal_ready));

        if (health.status === 'ok') {
          const model = await getModelStatus();
          if (!mounted) return;
          setModelLoaded(model.loaded && model.status === 'ready');
          if (model.multimodal_ready) {
            setMultimodalReady(true);
          }
        } else {
          setModelLoaded(false);
          setMultimodalReady(false);
        }
      } catch {
        if (!mounted) return;
        setBackendOnline(false);
        setModelLoaded(false);
        setMultimodalReady(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const renderPage = () => {
    switch (currentPage) {
      case 'overview':
        return <Overview />;
      case 'detection':
        return <SarcasmDetection />;
      case 'text-analysis':
        return <TextAnalysis />;
      case 'multimodal':
        return <MultimodalAnalysis />;
      case 'architecture':
        return <ModelArchitecture />;
      case 'performance':
        return <ModelPerformance />;
      case 'features':
        return <FeatureExtraction />;
      case 'dataset':
        return <Dataset />;
      case 'research':
        return <ResearchNovelty />;
      case 'about':
        return <AboutProject />;
      default:
        return <Overview />;
    }
  };

  return (
    <div className="flex min-h-screen bg-black text-zinc-100">
      {/* Collapsible Sidebar */}
      <Sidebar
        current={currentPage}
        onChange={setCurrentPage}
        backendOnline={backendOnline}
        modelLoaded={modelLoaded}
        multimodalReady={multimodalReady}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-black">
        <Header backendOnline={backendOnline} modelLoaded={modelLoaded} multimodalReady={multimodalReady} />

        <main className="flex-1 px-4 sm:px-8 py-6 max-w-7xl w-full mx-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentPage}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.15 }}
            >
              {renderPage()}
            </motion.div>
          </AnimatePresence>
        </main>

        <Footer />
      </div>
    </div>
  );
}
