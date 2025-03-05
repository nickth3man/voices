/**
 * Footer component for the Voices application
 * 
 * This component displays the application footer with copyright information.
 */

import React from 'react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="app-footer">
      <p>Voices Application &copy; {currentYear}</p>
    </footer>
  );
};

export default Footer;