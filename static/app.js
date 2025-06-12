// Mobile menu functionality
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const navLinks = document.getElementById('navLinks');

if (mobileMenuBtn && navLinks) {
  mobileMenuBtn.addEventListener('click', () => {
    navLinks.classList.toggle('active');
    const icon = mobileMenuBtn.querySelector('i');
    if (navLinks.classList.contains('active')) {
      icon.classList.remove('fa-bars');
      icon.classList.add('fa-times');
    } else {
      icon.classList.remove('fa-times');
      icon.classList.add('fa-bars');
    }
  });
}

// Close mobile menu when clicking outside
document.addEventListener('click', (e) => {
  if (navLinks && mobileMenuBtn && 
      !navLinks.contains(e.target) && 
      !mobileMenuBtn.contains(e.target)) {
    navLinks.classList.remove('active');
    const icon = mobileMenuBtn.querySelector('i');
    icon.classList.remove('fa-times');
    icon.classList.add('fa-bars');
  }
});

// Authentication modals
document.addEventListener('DOMContentLoaded', function() {
  const loginBtn = document.getElementById('login-btn');
  const signupBtn = document.getElementById('signup-btn');
  const loginModal = document.getElementById('login-modal');
  const signupModal = document.getElementById('signup-modal');
  const showSignupLink = document.getElementById('show-signup');
  const showLoginLink = document.getElementById('show-login');
  
  if (loginBtn && loginModal) {
    loginBtn.addEventListener('click', () => {
      loginModal.style.display = 'block';
    });
  }
  
  if (signupBtn && signupModal) {
    signupBtn.addEventListener('click', () => {
      signupModal.style.display = 'block';
    });
  }
  
  if (showSignupLink && signupModal && loginModal) {
    showSignupLink.addEventListener('click', () => {
      loginModal.style.display = 'none';
      signupModal.style.display = 'block';
    });
  }
  
  if (showLoginLink && loginModal && signupModal) {
    showLoginLink.addEventListener('click', () => {
      signupModal.style.display = 'none';
      loginModal.style.display = 'block';
    });
  }
  
  // Close modals when clicking outside
  window.addEventListener('click', (e) => {
    if (e.target === loginModal) {
      loginModal.style.display = 'none';
    }
    if (e.target === signupModal) {
      signupModal.style.display = 'none';
    }
  });
  
  // Emergency contacts
  const contactForm = document.getElementById('contact-form');
  if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const contactData = {
        name: document.getElementById('contact-name').value,
        type: document.getElementById('contact-type').value,
        phone: document.getElementById('contact-phone').value,
        email: document.getElementById('contact-email').value,
        notes: document.getElementById('contact-notes').value
      };
      
      fetch('/api/contacts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(contactData)
      })
      .then(response => response.json())
      .then(data => {
        // Add the new contact to the list
        addContactToList(data);
        // Reset the form
        contactForm.reset();
      })
      .catch(error => console.error('Error adding contact:', error));
    });
  }
  
  // Load emergency contacts
  function loadContacts() {
    fetch('/api/contacts')
      .then(response => response.json())
      .then(contacts => {
        const contactsList = document.getElementById('contacts-list');
        if (contactsList) {
          // Clear existing contacts
          contactsList.innerHTML = '';
          
          // Add each contact to the list
          contacts.forEach(contact => {
            addContactToList(contact);
          });
        }
      })
      .catch(error => console.error('Error loading contacts:', error));
  }
  
  // Add a contact to the list
  function addContactToList(contact) {
    const contactsList = document.getElementById('contacts-list');
    if (!contactsList) return;
    
    const contactCard = document.createElement('div');
    contactCard.className = 'contact-card';
    contactCard.dataset.id = contact.id;
    
    contactCard.innerHTML = `
      <div class="contact-header">
        <div class="contact-name">${contact.name}</div>
        <div class="contact-type">${contact.type}</div>
      </div>
      <div class="contact-details">
        <div><i class="fas fa-phone"></i> ${contact.phone}</div>
        ${contact.email ? `<div><i class="fas fa-envelope"></i> ${contact.email}</div>` : ''}
      </div>
      <div class="contact-actions">
        <button class="btn btn-outline btn-sm edit-contact" data-id="${contact.id}"><i class="fas fa-edit"></i> Edit</button>
        <button class="btn btn-outline btn-sm btn-danger delete-contact" data-id="${contact.id}"><i class="fas fa-trash"></i> Delete</button>
      </div>
    `;
    
    contactsList.appendChild(contactCard);
    
    // Add event listeners for edit and delete buttons
    const editBtn = contactCard.querySelector('.edit-contact');
    const deleteBtn = contactCard.querySelector('.delete-contact');
    
    if (editBtn) {
      editBtn.addEventListener('click', () => editContact(contact.id));
    }
    
    if (deleteBtn) {
      deleteBtn.addEventListener('click', () => deleteContact(contact.id));
    }
  }
  
  // Edit a contact
  function editContact(id) {
    // In a real app, this would open a modal with the contact's details
    // For now, we'll just log the ID
    console.log('Edit contact:', id);
  }
  
  // Delete a contact
  function deleteContact(id) {
    if (confirm('Are you sure you want to delete this contact?')) {
      fetch(`/api/contacts/${id}`, {
        method: 'DELETE'
      })
      .then(response => {
        if (response.ok) {
          // Remove the contact from the list
          const contactCard = document.querySelector(`.contact-card[data-id="${id}"]`);
          if (contactCard) {
            contactCard.remove();
          }
        }
      })
      .catch(error => console.error('Error deleting contact:', error));
    }
  }
  
  // Load contacts if on the emergency page
  if (document.querySelector('.emergency-page')) {
    loadContacts();
  }
  
  // Profile form submission
  const profileForm = document.getElementById('profile-form');
  if (profileForm) {
    profileForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const profileData = {
        name: document.getElementById('profile-name').value,
        dob: document.getElementById('profile-dob').value,
        email: document.getElementById('profile-email').value,
        phone: document.getElementById('profile-phone').value,
        address: document.getElementById('profile-address').value
      };
      
      fetch('/api/profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(profileData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Profile updated successfully');
      })
      .catch(error => console.error('Error updating profile:', error));
    });
  }
  
  // Health form submission
  const healthForm = document.getElementById('health-form');
  if (healthForm) {
    healthForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const healthData = {
        bloodType: document.getElementById('blood-type').value,
        genotype: document.getElementById('genotype').value,
        allergies: document.getElementById('allergies').value,
        conditions: document.getElementById('conditions').value
      };
      
      fetch('/api/health', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(healthData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Health information updated successfully');
      })
      .catch(error => console.error('Error updating health information:', error));
    });
  }
  
  // Settings form submissions
  const accountSettingsForm = document.getElementById('account-settings-form');
  if (accountSettingsForm) {
    accountSettingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const accountData = {
        email: document.getElementById('email').value,
        password: document.getElementById('password').value,
        confirmPassword: document.getElementById('confirm-password').value
      };
      
      if (accountData.password && accountData.password !== accountData.confirmPassword) {
        alert('Passwords do not match');
        return;
      }
      
      fetch('/api/account', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(accountData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Account settings updated successfully');
      })
      .catch(error => console.error('Error updating account settings:', error));
    });
  }
  
  const notificationSettingsForm = document.getElementById('notification-settings-form');
  if (notificationSettingsForm) {
    notificationSettingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const notificationData = {
        crisisAlerts: document.querySelector('input[name="crisis-alerts"]').checked,
        medicationReminders: document.querySelector('input[name="medication-reminders"]').checked,
        appointmentReminders: document.querySelector('input[name="appointment-reminders"]').checked,
        weeklyReports: document.querySelector('input[name="weekly-reports"]').checked,
        frequency: document.getElementById('notification-frequency').value
      };
      
      fetch('/api/notifications', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(notificationData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Notification settings updated successfully');
      })
      .catch(error => console.error('Error updating notification settings:', error));
    });
  }
  
  const privacySettingsForm = document.getElementById('privacy-settings-form');
  if (privacySettingsForm) {
    privacySettingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const privacyData = {
        shareHealthData: document.querySelector('input[name="share-health-data"]').checked,
        shareCrisisAlerts: document.querySelector('input[name="share-crisis-alerts"]').checked,
        shareAnonymousData: document.querySelector('input[name="share-anonymous-data"]').checked,
        dataRetention: document.getElementById('data-retention').value
      };
      
      fetch('/api/privacy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(privacyData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Privacy settings updated successfully');
      })
      .catch(error => console.error('Error updating privacy settings:', error));
    });
  }
  
  const deviceSettingsForm = document.getElementById('device-settings-form');
  if (deviceSettingsForm) {
    deviceSettingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const deviceData = {
        samplingRate: document.getElementById('sensor-sampling').value,
        batterySaver: document.getElementById('battery-saver').value,
        autoSync: document.querySelector('input[name="auto-sync"]').checked,
        backgroundMonitoring: document.querySelector('input[name="background-monitoring"]').checked
      };
      
      fetch('/api/device', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(deviceData)
      })
      .then(response => response.json())
      .then(data => {
        alert('Device settings updated successfully');
      })
      .catch(error => console.error('Error updating device settings:', error));
    });
  }
  
  // Data management actions
  const exportDataBtn = document.querySelector('.data-management-actions .btn-outline:nth-child(1)');
  if (exportDataBtn) {
    exportDataBtn.addEventListener('click', () => {
      fetch('/api/export-data')
        .then(response => response.blob())
        .then(blob => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'health-data.csv';
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          a.remove();
        })
        .catch(error => console.error('Error exporting data:', error));
    });
  }
  
  const clearDataBtn = document.querySelector('.data-management-actions .btn-outline:nth-child(2)');
  if (clearDataBtn) {
    clearDataBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear all your data? This action cannot be undone.')) {
        fetch('/api/clear-data', {
          method: 'POST'
        })
        .then(response => {
          if (response.ok) {
            alert('All data has been cleared');
          }
        })
        .catch(error => console.error('Error clearing data:', error));
      }
    });
  }
  
  const deleteAccountBtn = document.querySelector('.data-management-actions .btn-danger');
  if (deleteAccountBtn) {
    deleteAccountBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
        fetch('/api/delete-account', {
          method: 'POST'
        })
        .then(response => {
          if (response.ok) {
            window.location.href = '/';
          }
        })
        .catch(error => console.error('Error deleting account:', error));
      }
    });
  }
  
  // Sensor simulation
  const gsrInput = document.getElementById('gsrInput');
  const tempInput = document.getElementById('tempInput');
  const spo2Input = document.getElementById('spo2Input');
  const gsrInputValue = document.getElementById('gsrInputValue');
  const tempInputValue = document.getElementById('tempInputValue');
  const spo2InputValue = document.getElementById('spo2InputValue');
  
  if (gsrInput && tempInput && spo2Input) {
    gsrInput.addEventListener('input', () => {
      gsrInputValue.textContent = gsrInput.value;
    });
    
    tempInput.addEventListener('input', () => {
      tempInputValue.textContent = tempInput.value;
    });
    
    spo2Input.addEventListener('input', () => {
      spo2InputValue.textContent = spo2Input.value;
    });
    
    // Send data button
    const sendButton = document.getElementById('sendButton');
    if (sendButton) {
      sendButton.addEventListener('click', sendData);
    }
    
    // Auto send
    const autoButton = document.getElementById('autoButton');
    let autoInterval = null;
    
    if (autoButton) {
      autoButton.addEventListener('click', () => {
        if (autoInterval) {
          clearInterval(autoInterval);
          autoInterval = null;
          autoButton.textContent = 'Start Auto Send';
          autoButton.classList.remove('btn-primary');
          autoButton.classList.add('btn-outline');
        } else {
          autoInterval = setInterval(sendRandomData, 3000);
          autoButton.textContent = 'Stop Auto Send';
          autoButton.classList.remove('btn-outline');
          autoButton.classList.add('btn-primary');
        }
      });
    }
  }
  
  // Initialize Chart.js if on predictions page
  const chartCanvas = document.getElementById('timeSeriesChart');
  let chart;
  
  if (chartCanvas) {
    const ctx = chartCanvas.getContext('2d');
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'GSR',
            borderColor: 'rgb(255, 99, 132)',
            yAxisID: 'y',
            data: []
          },
          {
            label: 'Temperature',
            borderColor: 'rgb(255, 159, 64)',
            yAxisID: 'y1',
            data: []
          },
          {
            label: 'SpO2',
            borderColor: 'rgb(75, 192, 192)',
            yAxisID: 'y2',
            data: []
          },
          {
            label: 'Crisis Probability',
            borderColor: 'rgb(153, 102, 255)',
            yAxisID: 'y3',
            data: []
          }
        ]
      },
      options: {
        scales: {
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'GSR (µS)'
            },
            min: 0,
            max: 5
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Temp (°C)'
            },
            min: 35,
            max: 39,
            grid: {
              drawOnChartArea: false
            }
          },
          y2: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'SpO2 (%)'
            },
            min: 85,
            max: 100,
            grid: {
              drawOnChartArea: false
            }
          },
          y3: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Crisis Prob'
            },
            min: 0,
            max: 1,
            grid: {
              drawOnChartArea: false
            }
          }
        }
      }
    });
  }
  
  // Load database statistics
  loadStats();
  
  // Refresh stats button
  const refreshStatsButton = document.getElementById('refreshStats');
  if (refreshStatsButton) {
    refreshStatsButton.addEventListener('click', loadStats);
  }
  
  // Load recent data on startup
  fetch('/api/recent')
    .then(response => response.json())
    .then(data => {
      if (data.length > 0) {
        // Update with most recent reading
        const latest = data[data.length - 1];
        updateDashboard(
          {
            gsr: latest.gsr,
            temperature: latest.temperature,
            spo2: latest.spo2
          }, 
          {
            crisis_predicted: latest.crisis_predicted,
            crisis_probability: latest.crisis_probability
          }
        );
        
        // Update sliders
        if (gsrInput && tempInput && spo2Input) {
          gsrInput.value = latest.gsr;
          tempInput.value = latest.temperature;
          spo2Input.value = latest.spo2;
          gsrInputValue.textContent = latest.gsr.toFixed(1);
          tempInputValue.textContent = latest.temperature.toFixed(1);
          spo2InputValue.textContent = latest.spo2;
        }
      }
    })
    .catch(error => console.error('Error loading recent data:', error));
  
  // Functions
  function sendRandomData() {
    const gsr = Math.random() * 4.5 + 0.5;
    const temp = Math.random() * 4 + 35;
    const spo2 = Math.random() * 15 + 85;
    
    gsrInput.value = gsr.toFixed(1);
    tempInput.value = temp.toFixed(1);
    spo2Input.value = Math.round(spo2);
    
    gsrInputValue.textContent = gsr.toFixed(1);
    tempInputValue.textContent = temp.toFixed(1);
    spo2InputValue.textContent = Math.round(spo2);
    
    sendData();
  }
  
  function sendData() {
    const data = {
      gsr: parseFloat(gsrInput.value),
      temperature: parseFloat(tempInput.value),
      spo2: parseFloat(spo2Input.value)
    };
    
    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
      updateDashboard(data, result);
      // Refresh stats after sending data
      loadStats();
    })
    .catch(error => console.error('Error:', error));
  }
  
  function updateDashboard(data, result) {
    // Update readings
    const gsrValue = document.getElementById('gsrValue');
    const tempValue = document.getElementById('tempValue');
    const spo2Value = document.getElementById('spo2Value');
    const crisisRisk = document.getElementById('crisisRisk');
    const riskDescription = document.getElementById('riskDescription');
    
    if (gsrValue && tempValue && spo2Value) {
      gsrValue.textContent = data.gsr.toFixed(2);
      tempValue.textContent = data.temperature.toFixed(1);
      spo2Value.textContent = data.spo2;
    }
    
    // Update risk display
    if (crisisRisk && riskDescription) {
      const riskPercentage = (result.crisis_probability * 100).toFixed(1) + '%';
      crisisRisk.textContent = riskPercentage;
      
      if (result.crisis_predicted === 1) {
        crisisRisk.style.color = 'var(--danger)';
        riskDescription.textContent = 'CRISIS ALERT! Immediate preventive action required.';
      } else if (result.crisis_probability > 0.75) {
        crisisRisk.style.color = 'var(--danger)';
        riskDescription.textContent = 'Very high risk - Immediate preventive action required';
      } else if (result.crisis_probability > 0.5) {
        crisisRisk.style.color = 'var(--warning)';
        riskDescription.textContent = 'High risk - Take preventive measures and contact your healthcare provider';
      } else if (result.crisis_probability > 0.25) {
        crisisRisk.style.color = 'var(--warning)';
        riskDescription.textContent = 'Moderate risk - Increase hydration and rest';
      } else {
        crisisRisk.style.color = 'var(--success)';
        riskDescription.textContent = 'Low risk - Continue normal activities with standard precautions';
      }
    }
    
    // Update chart if it exists
    if (chart) {
      const timestamp = new Date().toLocaleTimeString();
      chart.data.labels.push(timestamp);
      chart.data.datasets[0].data.push(data.gsr);
      chart.data.datasets[1].data.push(data.temperature);
      chart.data.datasets[2].data.push(data.spo2);
      chart.data.datasets[3].data.push(result.crisis_probability);
      
      // Keep only the most recent 20 points
      if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(dataset => dataset.data.shift());
      }
      
      chart.update();
    }
  }
  
  function loadStats() {
    fetch('/api/stats')
      .then(response => response.json())
      .then(stats => {
        // Update statistics display
        const totalPredictions = document.getElementById('totalPredictions');
        const crisisPredictions = document.getElementById('crisisPredictions');
        const crisisPercentage = document.getElementById('crisisPercentage');
        const avgProbability = document.getElementById('avgProbability');
        const avgGsr = document.getElementById('avgGsr');
        const avgTemperature = document.getElementById('avgTemperature');
        const avgSpo2 = document.getElementById('avgSpo2');
        
        if (totalPredictions) totalPredictions.textContent = stats.total_predictions;
        if (crisisPredictions) crisisPredictions.textContent = stats.crisis_predictions;
        if (crisisPercentage) crisisPercentage.textContent = stats.crisis_percentage.toFixed(1) + '%';
        if (avgProbability) avgProbability.textContent = (stats.average_readings.crisis_probability * 100).toFixed(1) + '%';
        if (avgGsr) avgGsr.textContent = stats.average_readings.gsr.toFixed(2);
        if (avgTemperature) avgTemperature.textContent = stats.average_readings.temperature.toFixed(1) + '°C';
        if (avgSpo2) avgSpo2.textContent = stats.average_readings.spo2.toFixed(1) + '%';
      })
      .catch(error => console.error('Error loading stats:', error));
  }
  
  // Update crisis prediction
  function updatePrediction(probability) {
    const indicator = document.getElementById('crisis-indicator');
    const dot = indicator.querySelector('.indicator-dot');
    const text = indicator.querySelector('.indicator-text');
    const fill = document.getElementById('probability-fill');
    const probText = document.getElementById('probability-text');
    
    // Ensure probability is a valid number
    probability = parseFloat(probability) || 0;
    
    // Update probability meter
    const percentage = Math.min(Math.max(probability * 100, 0), 100);
    fill.style.width = `${percentage}%`;
    probText.textContent = `${percentage.toFixed(1)}%`;
    
    // Update indicator
    if (probability > 0.7) {
      dot.className = 'indicator-dot indicator-danger';
      text.textContent = `High Risk (${percentage.toFixed(1)}%)`;
      fill.style.backgroundColor = '#dc3545';
    } else if (probability > 0.4) {
      dot.className = 'indicator-dot indicator-warning';
      text.textContent = `Moderate Risk (${percentage.toFixed(1)}%)`;
      fill.style.backgroundColor = '#ffc107';
    } else {
      dot.className = 'indicator-dot indicator-normal';
      text.textContent = `Low Risk (${percentage.toFixed(1)}%)`;
      fill.style.backgroundColor = '#28a745';
    }
  }

  // Delete a medication
  async function handleDeleteMedication(medicationId, cardElement) {
    if (!confirm('Are you sure you want to delete this medication and all its history? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`/api/medications/${medicationId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Medication deleted:', result.message);
        if (cardElement) {
          cardElement.remove();
        }
        // Refresh current medications list (optional, if not all cards are always visible)
        // await loadCurrentMedications(); 
        
        // Refresh Today's Schedule
        await loadTodaysSchedule();
        
        // Refresh the medication history chart
        await loadMedicationHistory();
        
        // Optionally, show a success message to the user
        // showToast('Medication deleted successfully.'); 
      } else {
        const errorData = await response.json();
        console.error('Error deleting medication:', errorData.error);
        alert(`Error deleting medication: ${errorData.error}`);
      }
    } catch (error) {
      console.error('Network error or other issue:', error);
      alert('An error occurred while deleting the medication. Please check your connection and try again.');
    }
  }

  // --- Symptom Tracker Page Specific Functions ---
  if (document.querySelector('.symptom-tracker-page')) {
    const avgPainLevelEl = document.getElementById('avg-pain-level-value');
    const mostCommonSymptomEl = document.getElementById('most-common-symptom-value');
    const symptomFreeDaysEl = document.getElementById('symptom-free-days-value');
    const recentSymptomsListEl = document.getElementById('recent-symptoms-list');
    const saveSymptomsButton = document.getElementById('save-symptoms');
    const symptomDateField = document.getElementById('current-date'); // Assuming this shows the date being logged
    const painLevelSlider = document.getElementById('pain-level');
    const symptomNotesField = document.getElementById('symptom-notes');

    async function loadSymptomStats() {
      if (!avgPainLevelEl || !mostCommonSymptomEl || !symptomFreeDaysEl) return;
      try {
        const response = await fetch('/api/symptoms/stats');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const stats = await response.json();
        avgPainLevelEl.textContent = stats.average_pain_level !== null ? stats.average_pain_level : 'N/A';
        mostCommonSymptomEl.textContent = stats.most_common_symptom || 'N/A';
        symptomFreeDaysEl.textContent = stats.symptom_free_days;
      } catch (error) {
        console.error('Error loading symptom stats:', error);
        avgPainLevelEl.textContent = 'Error';
        mostCommonSymptomEl.textContent = 'Error';
        symptomFreeDaysEl.textContent = 'Error';
      }
    }

    function renderSymptomEntry(entry) {
      const card = document.createElement('div');
      card.className = 'symptom-card';
      
      const entryDate = new Date(entry.date + 'T00:00:00'); // Ensure date is parsed as local
      const formattedDate = entryDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

      let symptomsHTML = '';
      if (entry.symptoms && entry.symptoms.length > 0) {
        symptomsHTML = entry.symptoms.map(s => `<span class="symptom-tag">${s}</span>`).join('');
      } else {
        symptomsHTML = '<span class="symptom-tag-none">No specific symptoms listed</span>';
      }

      card.innerHTML = `
        <div class="symptom-header">
          <span class="date">${formattedDate}</span>
          <span class="pain-level">Pain Level: ${entry.pain_level}</span>
        </div>
        <div class="symptom-details">
          <div class="symptoms">
            ${symptomsHTML}
          </div>
          ${entry.notes ? `<p class="notes">${entry.notes}</p>` : '<p class="notes"><em>No notes.</em></p>'}
        </div>
      `;
      return card;
    }

    async function loadRecentSymptoms() {
      if (!recentSymptomsListEl) return;
      try {
        const response = await fetch('/api/symptoms');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const symptoms = await response.json();
        recentSymptomsListEl.innerHTML = ''; // Clear existing
        if (symptoms.length === 0) {
          recentSymptomsListEl.innerHTML = '<p>No symptoms logged yet.</p>';
          return;
        }
        symptoms.forEach(entry => {
          recentSymptomsListEl.appendChild(renderSymptomEntry(entry));
        });
      } catch (error) {
        console.error('Error loading recent symptoms:', error);
        recentSymptomsListEl.innerHTML = '<p>Error loading symptoms.</p>';
      }
    }

    if (saveSymptomsButton) {
      saveSymptomsButton.addEventListener('click', async () => {
        // Assuming symptomDateField.textContent or a hidden input holds the YYYY-MM-DD date
        // For simplicity, this example might need a robust way to get the selected log date
        // For now, let's assume a hidden input or a parsable field for the date.
        // This part needs to align with how the date is actually selected and stored in the form.
        // Let's use a placeholder for date retrieval from the form, assuming it is YYYY-MM-DD
        // const logDate = document.getElementById('actual-log-date-input').value; // Example: needs this element
        
        // Fallback: Try to parse from the displayed #current-date span if it contains YYYY-MM-DD
        // This is brittle; a dedicated hidden input for the selected YYYY-MM-DD date is better.
        let logDateStr = new Date().toISOString().split('T')[0]; // Default to today if parsing fails
        if(symptomDateField) {
            // Attempt to parse a date from a more complex string if needed.
            // For now, assuming symptomDateField holds or can be converted to YYYY-MM-DD
            // This part is highly dependent on how '#current-date' is formatted and updated.
            // We will assume for this example it holds a date that can be directly used or needs parsing.
            // If it's like "Today, April 17, 2024", it needs robust parsing not included here.
            // For the sake of an example, let's assume you have a mechanism to get YYYY-MM-DD.
            // If symptomDateField.dataset.isoDate exists (set by date picker logic), use it.
            if (symptomDateField.dataset.isoDate) {
                logDateStr = symptomDateField.dataset.isoDate;
            } else {
                // Simple attempt if it's just a YYYY-MM-DD string, otherwise defaults to today
                // This part is NOT robust for display strings like "Today, April 17, 2024"
                // You should ensure `logDateStr` is correctly set to a 'YYYY-MM-DD' string.
                console.warn("Using default/today's date for symptom log as specific date parsing from display is complex/not implemented here.");
            }
        }

        const selectedSymptoms = [];
        document.querySelectorAll('.symptom-checklist input[name="symptoms"]:checked').forEach(checkbox => {
          selectedSymptoms.push(checkbox.value);
        });

        const payload = {
          date: logDateStr, // Ensure this is YYYY-MM-DD format
          pain_level: parseInt(painLevelSlider.value, 10),
          symptoms: selectedSymptoms,
          notes: symptomNotesField.value
        };

        try {
          const response = await fetch('/api/symptoms', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
          });
          if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || `HTTP error! status: ${response.status}`);
          }
          alert('Symptoms saved successfully!');
          // Clear form elements (optional, good UX)
          painLevelSlider.value = '0';
          document.querySelectorAll('.symptom-checklist input[name="symptoms"]:checked').forEach(checkbox => checkbox.checked = false);
          symptomNotesField.value = '';
          
          // Refresh stats and list
          loadSymptomStats();
          loadRecentSymptoms();
        } catch (error) {
          console.error('Error saving symptoms:', error);
          alert(`Error saving symptoms: ${error.message}`);
        }
      });
    }

    // Initial load
    loadSymptomStats();
    loadRecentSymptoms();
    
    // Add event listener for date changes if your date selector is interactive
    // Example: if #prev-day, #next-day, or a date picker updates symptomDateField.dataset.isoDate,
    // you might want to reload data or clear the form for the new date.
    // The current save logic assumes it can get the correct YYYY-MM-DD date for the log.
  }
  // --- End Symptom Tracker Page Specific Functions ---

  // Caregiver dashboard functionality
  const patientDetailsModal = document.getElementById('patientDetailsModal');
  const closeModalBtn = patientDetailsModal ? patientDetailsModal.querySelector('.close') : null;
  let vitalSignsChart = null;

  function formatTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function updateChart(canvasId, readings) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    if (vitalSignsChart) {
      vitalSignsChart.destroy();
    }

    if (!readings || readings.length === 0) {
      document.querySelector('.chart-empty-message').style.display = 'block';
      document.getElementById(canvasId).style.display = 'none';
      return;
    }
    
    document.querySelector('.chart-empty-message').style.display = 'none';
    document.getElementById(canvasId).style.display = 'block';

    vitalSignsChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: readings.map(r => formatTime(r.timestamp)),
        datasets: [
          {
            label: 'Temperature (°C)',
            data: readings.map(r => r.temperature),
            borderColor: '#ff6384',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            yAxisID: 'y',
          },
          {
            label: 'SpO2 (%)',
            data: readings.map(r => r.spo2),
            borderColor: '#36a2eb',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            yAxisID: 'y1',
          },
          {
            label: 'GSR',
            data: readings.map(r => r.gsr),
            borderColor: '#4bc0c0',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            yAxisID: 'y2',
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: { display: true, text: 'Temp (°C)' }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: { display: true, text: 'SpO2 (%)' },
            grid: { drawOnChartArea: false }
          },
          y2: {
            type: 'linear',
            display: true,
            position: 'right',
            title: { display: true, text: 'GSR' },
            grid: { drawOnChartArea: false },
            ticks: { display: false }
          }
        }
      }
    });
  }

  async function openPatientDetailsModal(patientId) {
    try {
      const response = await fetch(`/api/patient/${patientId}/details`);
      if (!response.ok) {
        throw new Error('Failed to fetch patient details');
      }
      const data = await response.json();

      // Populate patient info
      const patientInfoDiv = patientDetailsModal.querySelector('.patient-details');
      patientInfoDiv.innerHTML = `
        <h3>Patient Info</h3>
        <p><strong>Name:</strong> ${data.patient.name}</p>
        <p><strong>Email:</strong> ${data.patient.email}</p>
        <p><strong>Age:</strong> ${data.patient.age || 'N/A'}</p>
        <p><strong>Gender:</strong> ${data.patient.gender || 'N/A'}</p>
      `;

      // Populate medication schedule
      const medicationDiv = patientDetailsModal.querySelector('.medication-schedule');
      let medicationHtml = '<h3>Today\'s Medication</h3>';
      if (data.medication_schedule.length > 0) {
        medicationHtml += '<ul>';
        data.medication_schedule.forEach(med => {
          medicationHtml += `<li>${med.medication_name} (${med.dosage}) at ${med.time} - ${med.is_taken ? 'Taken' : 'Pending'}</li>`;
        });
        medicationHtml += '</ul>';
      } else {
        medicationHtml += '<p>No medications scheduled for today.</p>';
      }
      medicationDiv.innerHTML = medicationHtml;

      // Populate recent symptoms
      const symptomsDiv = patientDetailsModal.querySelector('.recent-symptoms');
      let symptomsHtml = '<h3>Recent Symptoms</h3>';
      if (data.recent_symptoms.length > 0) {
        symptomsHtml += '<ul>';
        data.recent_symptoms.forEach(symptom => {
          symptomsHtml += `<li>${symptom.date}: Pain Level ${symptom.pain_level}, Symptoms: ${symptom.symptoms}</li>`;
        });
        symptomsHtml += '</ul>';
      } else {
        symptomsHtml += '<p>No recent symptoms logged.</p>';
      }
      symptomsDiv.innerHTML = symptomsHtml;

      // Update chart
      updateChart('vitalSignsChart', data.sensor_readings);

      patientDetailsModal.style.display = 'block';
    } catch (error) {
      console.error('Error opening patient details modal:', error);
      alert('Could not load patient details. Please try again.');
    }
  }

  // Event listener for all "View Details" buttons
  document.body.addEventListener('click', function(event) {
    if (event.target.matches('.view-details-btn')) {
      const patientId = event.target.dataset.patientId;
      openPatientDetailsModal(patientId);
    }
  });

  // Close modal
  if (closeModalBtn) {
    closeModalBtn.onclick = function() {
      patientDetailsModal.style.display = "none";
    }
  }

  window.onclick = function(event) {
    if (event.target == patientDetailsModal) {
      patientDetailsModal.style.display = "none";
    }
  }

  // Contact Patient Modal
  const contactModal = document.getElementById('contact-patient-modal');
  const closeContactModalBtn = document.getElementById('close-contact-modal');

  document.body.addEventListener('click', function(event) {
    if (event.target.matches('.contact-btn')) {
      const patientName = event.target.dataset.patientName;
      const patientEmail = event.target.dataset.patientEmail;
      const patientPhone = event.target.dataset.patientPhone;

      document.getElementById('contact-modal-title').textContent = `Contact ${patientName}`;
      document.getElementById('contact-patient-name').textContent = patientName;
      document.getElementById('contact-patient-email').textContent = patientEmail;
      document.getElementById('contact-patient-email').href = `mailto:${patientEmail}`;
      document.getElementById('contact-patient-phone').textContent = patientPhone;
      document.getElementById('contact-patient-phone').href = `tel:${patientPhone}`;

      document.getElementById('email-patient-link').href = `mailto:${patientEmail}`;
      document.getElementById('call-patient-link').href = `tel:${patientPhone}`;
      
      contactModal.style.display = 'block';
    }
  });

  if (closeContactModalBtn) {
    closeContactModalBtn.onclick = function() {
      contactModal.style.display = 'none';
    }
  }

  window.addEventListener('click', (event) => {
    if (event.target == contactModal) {
      contactModal.style.display = 'none';
    }
  });

  // "Add Patient" Modal functionality
  const addPatientModal = document.getElementById('addPatientModal');
  const openAddPatientBtn = document.getElementById('add-patient-btn-main');
  const closeAddPatientBtn = document.getElementById('closeAddPatientModal');
  const addPatientForm = document.getElementById('addPatientForm');
  const linkMethodRadios = document.querySelectorAll('input[name="linkMethod"]');
  const emailField = document.getElementById('emailField');
  const otpField = document.getElementById('otpField');

  if (openAddPatientBtn) {
    openAddPatientBtn.addEventListener('click', () => {
      addPatientModal.style.display = 'block';
    });
  }

  if (closeAddPatientBtn) {
    closeAddPatientBtn.addEventListener('click', () => {
      addPatientModal.style.display = 'none';
    });
  }

  linkMethodRadios.forEach(radio => {
    radio.addEventListener('change', () => {
      if (radio.value === 'email') {
        emailField.style.display = 'block';
        otpField.style.display = 'none';
      } else {
        emailField.style.display = 'none';
        otpField.style.display = 'block';
      }
    });
  });

  if (addPatientForm) {
    addPatientForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(addPatientForm);
      const linkMethod = formData.get('linkMethod');
      const relationship = document.getElementById('relationship').value;
      const patientEmail = document.getElementById('patientEmail').value;
      const patientOTP = document.getElementById('patientOTP').value;

      const body = {
        linkMethod,
        relationship,
        email: linkMethod === 'email' ? patientEmail : undefined,
        otp: linkMethod === 'otp' ? patientOTP : undefined
      };

      try {
        const response = await fetch('/api/caregiver/link-patient', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });

        const result = await response.json();

        if (response.ok && result.success) {
          alert(result.message);
          addPatientModal.style.display = 'none';
          location.reload(); // Reload the page to show the new patient
        } else {
          throw new Error(result.message || 'Failed to link patient.');
        }
      } catch (error) {
        console.error('Error linking patient:', error);
        alert(error.message);
      }
    });
  }
});
