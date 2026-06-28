import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "AIzaSyA20rbt7u4z311_5P498mFOXEGgVMexgOM",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "deepfake-detection-syste-10a8b.firebaseapp.com",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "deepfake-detection-syste-10a8b",
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "deepfake-detection-syste-10a8b.firebasestorage.app",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "701823507922",
  appId: import.meta.env.VITE_FIREBASE_APP_ID || "1:701823507922:web:6d2f097b235e192bb6bc0b"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Cloud Firestore and get a reference to the service
export const db = getFirestore(app);
