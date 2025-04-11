# enhancements.py - Modular Enhancements for app.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import pandas as pd
import speech_recognition as sr
import sqlite3
from app import get_db_path, get_transactions, get_budget
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans

# ------------------------------
# 📧 Email Notification System
# ------------------------------

EMAIL_ADDRESS = "your_email@gmail.com"  # replace with your email
EMAIL_PASSWORD = "your_app_password"    # use app password if using Gmail

def send_budget_email(user_email, username, user_id):
    try:
        df = get_transactions(user_id)
        budget = get_budget(user_id)

        if df.empty:
            body = f"Hi {username},\n\nYou have no transactions this month.\n\nBest,\nExpense Tracker"
        else:
            this_month = datetime.now().strftime('%Y-%m')
            df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
            this_month_data = df[df['month'] == this_month]
            total_expense = this_month_data[this_month_data['type'] == 'expense']['amount'].sum()
            total_income = this_month_data[this_month_data['type'] == 'income']['amount'].sum()

            body = f"""
            Hi {username},

            Here's your monthly expense summary:

            Total Income: Rs{total_income:.2f}
            Total Expenses: Rs{total_expense:.2f}
            Budget Limit: Rs{budget:.2f if budget else 0}

            """
            if budget and total_expense > budget:
                body += f"⚠️ You've exceeded your budget by Rs{total_expense - budget:.2f}\n"
                # Include investment recommendations in email
                investment_tips = generate_investment_recommendations(user_id)
                body += f"\n💡 Investment Recommendations:\n{investment_tips}\n"
            elif budget:
                body += f"✅ You're within budget. Remaining: Rs{budget - total_expense:.2f}\n"
            body += "\nBest,\nExpense Tracker"

        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = user_email
        msg['Subject'] = "📊 Your Monthly Expense Summary"

        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True, "Email sent successfully."
    except Exception as e:
        return False, str(e)

# ---------------------------------
# 💡 Budget Recommendation System
# ---------------------------------

def get_budget_recommendation(user_id):
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ?", conn, params=(user_id,))
    conn.close()

    if df.empty:
        return "Not enough data to recommend a budget."

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df_expense = df[df['type'] == 'expense']

    monthly_avg = df_expense.groupby('month')['amount'].sum().mean()
    return f"Based on past data, your recommended budget is Rs{monthly_avg:.2f}"

# ----------------------------------
# 🎙️ Voice Expense Entry Assistant
# ----------------------------------

def listen_and_parse_expense():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening for expense entry...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        # Simple format: "Add 200 groceries today"
        tokens = command.lower().split()
        amount = float(tokens[tokens.index("add") + 1])
        category = tokens[tokens.index("add") + 2]
        date = datetime.now().strftime('%Y-%m-%d')
        return {
            'amount': amount,
            'category': category.title(),
            'date': date,
            'type': 'expense'
        }
    except Exception as e:
        print("Voice input error:", e)
        return None

# --------------------------------------
# 💹 AI Investment Recommendations
# --------------------------------------

def get_spending_pattern(user_id):
    """Analyze user spending patterns to identify investment opportunities"""
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ? AND type = 'expense'", 
                          conn, params=(user_id,))
    conn.close()
    
    if df.empty or len(df) < 5:
        return None, "Not enough transaction data to analyze spending patterns."
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate monthly spending by category
    df['month'] = df['date'].dt.to_period('M')
    monthly_category = df.groupby(['month', 'category'])['amount'].sum().reset_index()
    
    # Calculate spending volatility by category (standard deviation)
    volatility = monthly_category.groupby('category')['amount'].std().reset_index()
    volatility.columns = ['category', 'volatility']
    
    # Calculate average monthly spend by category
    avg_spend = monthly_category.groupby('category')['amount'].mean().reset_index()
    avg_spend.columns = ['category', 'avg_monthly']
    
    # Merge the data
    spending_pattern = pd.merge(avg_spend, volatility, on='category', how='left')
    spending_pattern['volatility'] = spending_pattern['volatility'].fillna(0)
    
    return spending_pattern, "Success"

def generate_investment_recommendations(user_id):
    """Generate personalized investment recommendations based on spending patterns"""
    spending_pattern, msg = get_spending_pattern(user_id)
    
    if spending_pattern is None:
        return "We need more transaction data to provide personalized investment recommendations."
    
    # Get current budget and income
    conn = sqlite3.connect(get_db_path())
    budget = get_budget(user_id)
    
    df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ? AND type = 'income'", 
                          conn, params=(user_id,))
    conn.close()
    
    if df.empty:
        avg_income = 0
    else:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        avg_income = df.groupby('month')['amount'].sum().mean()
    
    # Get total monthly expenses
    total_monthly_expense = spending_pattern['avg_monthly'].sum()
    
    # Calculate potential savings
    potential_monthly_savings = avg_income - total_monthly_expense if avg_income > 0 else 0
    
    # Identify discretionary spending (categories with high volatility)
    discretionary = spending_pattern.sort_values('volatility', ascending=False).head(3)
    
    # Generate recommendations
    recommendations = []
    
    # If savings potential is identified
    if potential_monthly_savings > 0:
        recommendations.append(f"Based on your income and spending patterns, you could save approximately Rs{potential_monthly_savings:.2f} per month.")
        
        # Recommend investment allocation based on saving amount
        if potential_monthly_savings < 1000:
            recommendations.append("Consider starting with a recurring deposit or a liquid fund for your savings.")
        elif potential_monthly_savings < 5000:
            recommendations.append("Consider allocating 70% to a fixed deposit and 30% to a low-risk mutual fund.")
        else:
            recommendations.append("Consider a diversified investment approach: 50% in fixed deposits, 30% in mutual funds, and 20% in a tax-saving instrument like ELSS.")
    
    # If over budget, suggest expense reduction
    if budget and total_monthly_expense > budget:
        excess = total_monthly_expense - budget
        
        if not discretionary.empty:
            highest_category = discretionary.iloc[0]['category']
            highest_amount = discretionary.iloc[0]['avg_monthly']
            
            recommendations.append(f"You could reduce spending in '{highest_category}' (avg. Rs{highest_amount:.2f}/month) to stay within your budget.")
    
    # Default recommendation if nothing specific
    if not recommendations:
        recommendations.append("Consider setting up an emergency fund with 3-6 months of expenses before investing in market-linked products.")
        recommendations.append("For long-term goals, a mix of mutual funds, fixed deposits, and government schemes like PPF can provide balanced growth.")
    
    return "\n- ".join([""] + recommendations)

# ----------------------------------
# 🚨 Budget Alert System
# ----------------------------------

def check_budget_status(user_id):
    """Check if user is approaching or has exceeded budget limits"""
    budget = get_budget(user_id)
    
    if not budget:
        return None, "No budget set"
    
    # Get current month transactions
    today = datetime.now()
    start_of_month = today.replace(day=1).strftime('%Y-%m-%d')
    end_of_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    end_of_month = end_of_month.strftime('%Y-%m-%d')
    
    df = get_transactions(user_id, start_of_month, end_of_month)
    
    if df.empty:
        return None, "No transactions this month"
    
    # Calculate current month expenses
    expenses = df[df['type'] == 'expense']['amount'].sum()
    
    # Calculate percentage of budget used
    budget_used_pct = (expenses / budget) * 100
    
    status = {
        "budget": budget,
        "expenses": expenses,
        "remaining": budget - expenses,
        "percentage_used": budget_used_pct,
        "exceeded": expenses > budget
    }
    
    return status, "Success"

def show_budget_alert():
    """Display a budget alert if user has exceeded their budget"""
    if not st.session_state.get('user_id'):
        return
    
    user_id = st.session_state['user_id']
    
    # Check if we've already shown the alert in this session
    if st.session_state.get('budget_alert_shown'):
        return
    
    status, msg = check_budget_status(user_id)
    
    if status is None or not status.get('exceeded', False):
        return
    
    # Mark that we've shown the alert to avoid repeated popups
    st.session_state['budget_alert_shown'] = True
    
    # Generate investment recommendations
    recommendations = generate_investment_recommendations(user_id)
    
    # Show alert
    st.warning(f"⚠️ Budget Alert: You've exceeded your monthly budget by Rs{status['expenses'] - status['budget']:.2f}")
    
    with st.expander("💡 See saving and investment recommendations"):
        st.markdown("### How to get back on track:")
        st.markdown(recommendations)
        
        # Add action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Set up automatic savings"):
                st.session_state['show_savings_setup'] = True
        with col2:
            if st.button("View detailed spending analysis"):
                st.session_state['show_spending_analysis'] = True

# ----------------------------------
# 📊 Spending Analysis
# ----------------------------------

def analyze_spending(user_id):
    """Perform detailed spending analysis for budget optimization"""
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ? AND type = 'expense'", 
                          conn, params=(user_id,))
    conn.close()
    
    if df.empty or len(df) < 5:
        return None, "Not enough transaction data for analysis."
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()
    
    # Create analysis results
    analysis = {
        "by_category": df.groupby('category')['amount'].agg(['sum', 'mean', 'count']).reset_index(),
        "by_day": df.groupby('day_of_week')['amount'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
        "by_month": df.groupby('month')['amount'].sum(),
        "recent_trend": df.set_index('date').resample('W')['amount'].sum().tail(8)
    }
    
    # Try to apply KMeans clustering to find spending patterns if enough data
    if len(df) >= 20:
        try:
            # Prepare data for clustering
            numeric_data = pd.get_dummies(df[['category', 'day_of_week']])
            numeric_data['amount'] = df['amount']
            
            # Normalize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=min(3, len(df) // 7), random_state=42)
            df['cluster'] = kmeans.fit_predict(scaled_data)
            
            # Get insights from clusters
            cluster_insights = df.groupby('cluster').agg({
                'amount': ['mean', 'sum'],
                'category': lambda x: pd.Series.mode(x)[0],
                'day_of_week': lambda x: pd.Series.mode(x)[0]
            })
            
            analysis["spending_patterns"] = cluster_insights
        except Exception as e:
            print(f"Error in KMeans clustering: {e}")
    
    return analysis, "Success"

def show_spending_analysis(user_id):
    """Display spending analysis in the Streamlit app"""
    if st.session_state.get('show_spending_analysis'):
        analysis, msg = analyze_spending(user_id)
        
        if analysis is None:
            st.info(msg)
            return
        
        st.subheader("📊 Spending Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Expense Categories")
            top_categories = analysis["by_category"].sort_values(by='sum', ascending=False).head(5)
            for i, row in top_categories.iterrows():
                st.metric(
                    label=row['category'], 
                    value=f"Rs{row['sum']:.2f}", 
                    delta=f"{row['count']} transactions"
                )
                
        with col2:
            st.markdown("### Spending by Day of Week")
            day_data = analysis["by_day"].reset_index()
            day_data.columns = ['Day', 'Amount']
            st.bar_chart(day_data.set_index('Day'))
        
        st.markdown("### Recent Weekly Spending Trend")
        trend_data = analysis["recent_trend"].reset_index()
        trend_data.columns = ['Week', 'Amount']
        st.line_chart(trend_data.set_index('Week'))
        
        # Show optimization suggestions
        st.subheader("💡 Optimization Suggestions")
        
        # Find the highest spending day
        highest_day = analysis["by_day"].idxmax()
        highest_day_amount = analysis["by_day"].max()
        
        # Find the highest category
        highest_category = analysis["by_category"].sort_values(by='sum', ascending=False).iloc[0]
        
        suggestions = [
            f"Your highest spending day is **{highest_day}** (Rs{highest_day_amount:.2f}). Consider planning ahead for this day.",
            f"Your largest expense category is **{highest_category['category']}** (Rs{highest_category['sum']:.2f}). Look for ways to reduce spending here."
        ]
        
        # Add pattern-based suggestions if available
        if "spending_patterns" in analysis:
            for i, pattern in enumerate(analysis["spending_patterns"].iterrows()):
                cluster_id, data = pattern
                category = data[('category', '<lambda>')]
                day = data[('day_of_week', '<lambda>')]
                amount = data[('amount', 'mean')]
                
                suggestions.append(f"You tend to spend Rs{amount:.2f} on **{category}** on **{day}**.")
            
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
            
        # Reset the state variable if the user dismisses the analysis
        if st.button("Close Analysis"):
            st.session_state['show_spending_analysis'] = False
            st.rerun()

# ----------------------------------
# 💵 Automatic Savings Setup
# ----------------------------------

def show_savings_setup(user_id):
    """Display interface for setting up automatic savings rules"""
    if st.session_state.get('show_savings_setup'):
        st.subheader("💵 Automatic Savings Setup")
        
        st.info("Set up rules to automatically track your savings goals based on your spending habits.")
        
        # Get user income data
        conn = sqlite3.connect(get_db_path())
        income_df = pd.read_sql_query("SELECT * FROM transactions WHERE user_id = ? AND type = 'income'", 
                                      conn, params=(user_id,))
        conn.close()
        
        if income_df.empty:
            st.warning("We need income information to suggest savings plans. Please add your income transactions first.")
            return
        
        # Calculate average monthly income
        income_df['date'] = pd.to_datetime(income_df['date'])
        income_df['month'] = income_df['date'].dt.to_period('M')
        avg_monthly_income = income_df.groupby('month')['amount'].sum().mean()
        
        st.metric("Average Monthly Income", f"Rs{avg_monthly_income:.2f}")
        
        # Suggest savings percentages
        st.markdown("### Suggested Savings Plan")
        st.markdown("Based on the 50/30/20 rule:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Essentials (50%)", f"Rs{avg_monthly_income * 0.5:.2f}")
        with col2:
            st.metric("Wants (30%)", f"Rs{avg_monthly_income * 0.3:.2f}")
        with col3:
            st.metric("Savings (20%)", f"Rs{avg_monthly_income * 0.2:.2f}")
        
        # Allow user to set their own goals
        st.markdown("### Set Your Custom Savings Goal")
        custom_goal_pct = st.slider("Percentage of income to save", 5, 50, 20)
        custom_goal_amount = (avg_monthly_income * custom_goal_pct / 100)
        
        # Investment allocation suggestion
        st.markdown("### Recommended Investment Allocation")
        
        # Adjust recommendations based on savings amount
        if custom_goal_amount < 1000:
            emergency_pct = 100
            fixed_deposit_pct = 0
            mutual_fund_pct = 0
        elif custom_goal_amount < 5000:
            emergency_pct = 60
            fixed_deposit_pct = 40
            mutual_fund_pct = 0
        else:
            emergency_pct = 40
            fixed_deposit_pct = 40
            mutual_fund_pct = 20
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Emergency Fund", f"{emergency_pct}%")
            emergency_amount = custom_goal_amount * emergency_pct / 100
            st.text(f"Rs{emergency_amount:.2f}/month")
        with col2:
            st.metric("Fixed Deposits", f"{fixed_deposit_pct}%")
            fd_amount = custom_goal_amount * fixed_deposit_pct / 100
            st.text(f"Rs{fd_amount:.2f}/month")
        with col3:
            st.metric("Mutual Funds", f"{mutual_fund_pct}%")
            mf_amount = custom_goal_amount * mutual_fund_pct / 100
            st.text(f"Rs{mf_amount:.2f}/month")
        
        # Save settings
        if st.button("Save Savings Plan"):
            # Here you would save these settings to your database
            # For now we'll just store in session state
            st.session_state['savings_plan'] = {
                "goal_percentage": custom_goal_pct,
                "monthly_amount": custom_goal_amount,
                "allocations": {
                    "emergency_fund": emergency_pct,
                    "fixed_deposit": fixed_deposit_pct,
                    "mutual_fund": mutual_fund_pct
                }
            }
            st.success("Savings plan saved! We'll track your progress towards these goals.")
            # Add a hook into the main app to track these goals
            
        # Reset the state variable
        if st.button("Close"):
            st.session_state['show_savings_setup'] = False
            st.rerun()

# ----------------------------------
# 🚀 Feature Flag System
# ----------------------------------

# Default feature flags
DEFAULT_FEATURE_FLAGS = {
    "ai_investment_recommendations": True,
    "budget_alerts": True,
    "spending_analysis": True,
    "auto_savings": True,
    "voice_expense_entry": True,
    "email_notifications": True
}

def init_feature_flags():
    """Initialize feature flags in the database"""
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    # Create feature_flags table if not exists
    c.execute('''
    CREATE TABLE IF NOT EXISTS feature_flags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        enabled INTEGER DEFAULT 1,
        description TEXT,
        rollout_percentage INTEGER DEFAULT 100
    )
    ''')
    
    # Insert default feature flags if not exist
    for flag_name, enabled in DEFAULT_FEATURE_FLAGS.items():
        description = " ".join(word.capitalize() for word in flag_name.split('_'))
        
        c.execute("SELECT id FROM feature_flags WHERE name = ?", (flag_name,))
        if not c.fetchone():
            c.execute(
                "INSERT INTO feature_flags (name, enabled, description, rollout_percentage) VALUES (?, ?, ?, ?)",
                (flag_name, 1 if enabled else 0, description, 100)
            )
    
    conn.commit()
    conn.close()

def is_feature_enabled(feature_name, user_id=None):
    """Check if a feature is enabled for a specific user"""
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    c.execute("SELECT enabled, rollout_percentage FROM feature_flags WHERE name = ?", (feature_name,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return DEFAULT_FEATURE_FLAGS.get(feature_name, False)
    
    enabled, rollout_percentage = result
    
    if not enabled:
        return False
    
    # If 100% rollout, enable for everyone
    if rollout_percentage >= 100:
        return True
    
    # Deterministic user-based percentage rollout
    if user_id:
        # Use hash of feature name + user_id for deterministic distribution
        import hashlib
        hash_input = f"{feature_name}:{user_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        user_bucket = hash_value % 100
        
        return user_bucket < rollout_percentage
    
    return False

def get_all_feature_flags():
    """Get all feature flags and their status"""
    conn = sqlite3.connect(get_db_path())
    query = "SELECT name, enabled, description, rollout_percentage FROM feature_flags"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def update_feature_flag(flag_name, enabled, rollout_percentage=None):
    """Update feature flag settings"""
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    if rollout_percentage is not None:
        c.execute(
            "UPDATE feature_flags SET enabled = ?, rollout_percentage = ? WHERE name = ?",
            (1 if enabled else 0, rollout_percentage, flag_name)
        )
    else:
        c.execute(
            "UPDATE feature_flags SET enabled = ? WHERE name = ?",
            (1 if enabled else 0, flag_name)
        )
    
    conn.commit()
    conn.close()
    return True

# Function to add to your app's startup to initialize feature flags
def initialize_features():
    """Initialize all feature-related systems"""
    init_feature_flags()

# Functions to be called from the main app
def add_feature_flag_management_to_admin_panel():
    """Add feature flag management UI to the admin panel"""
    st.subheader("Feature Flag Management")
    
    # Get all feature flags
    flags_df = get_all_feature_flags()
    
    if flags_df.empty:
        st.info("No feature flags defined.")
        return
    
    # Display each feature flag with controls
    for _, flag in flags_df.iterrows():
        col1, col2, col3 = st.columns([3, 1, 2])
        
        with col1:
            st.markdown(f"### {flag['description']}")
            st.text(f"Feature ID: {flag['name']}")
        
        with col2:
            enabled = st.checkbox("Enabled", value=bool(flag['enabled']), key=f"flag_{flag['name']}")
            
            if enabled != bool(flag['enabled']):
                update_feature_flag(flag['name'], enabled)
                st.rerun()
        
        with col3:
            rollout = st.slider(
                "Rollout %", 
                min_value=0, 
                max_value=100, 
                value=int(flag['rollout_percentage']),
                key=f"rollout_{flag['name']}"
            )
            
            if rollout != int(flag['rollout_percentage']):
                update_feature_flag(flag['name'], enabled, rollout)
                st.rerun()
        
        st.divider()

# Example of how to use feature flags in the main app:
def run_feature_controlled_enhancements():
    """Run all enhancements based on feature flag settings"""
    if not st.session_state.get('user_id'):
        return
    
    user_id = st.session_state['user_id']
    
    # Budget alerts feature
    if is_feature_enabled('budget_alerts', user_id):
        show_budget_alert()
    
    # Spending analysis feature
    if is_feature_enabled('spending_analysis', user_id):
        show_spending_analysis(user_id)
    
    # Auto savings feature
    if is_feature_enabled('auto_savings', user_id):
        show_savings_setup(user_id)
