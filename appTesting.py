"""
Test Document Generator for Smart Document Organizer
Creates various test documents with different similarity levels
Run this script to generate .txt files in a 'test_documents' folder
"""

import os

# Create test_documents directory
os.makedirs('test_documents', exist_ok=True)

# ============================================================================
# SET 1: EXACT DUPLICATES (100% Similarity)
# ============================================================================

doc1_original = """
Artificial intelligence and machine learning are revolutionizing the technology industry. 
These advanced computational systems can process vast amounts of data and identify complex 
patterns that humans might miss. Deep learning algorithms, powered by neural networks, have 
achieved remarkable success in image recognition, natural language processing, and autonomous 
systems. Companies worldwide are investing billions of dollars in AI research and development 
to gain competitive advantages in their respective markets.
"""

doc1_duplicate = """
Artificial intelligence and machine learning are revolutionizing the technology industry. 
These advanced computational systems can process vast amounts of data and identify complex 
patterns that humans might miss. Deep learning algorithms, powered by neural networks, have 
achieved remarkable success in image recognition, natural language processing, and autonomous 
systems. Companies worldwide are investing billions of dollars in AI research and development 
to gain competitive advantages in their respective markets.
"""

# ============================================================================
# SET 2: HIGH SIMILARITY (85-95% Similar - Different wording, same content)
# ============================================================================

doc2_original = """
Climate change represents one of the most significant challenges facing humanity today. 
Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather 
patterns to become increasingly unpredictable. Scientists warn that without immediate action 
to reduce carbon emissions, we may face catastrophic consequences including extreme droughts, 
devastating floods, and mass species extinction. Governments and organizations must work 
together to implement sustainable policies and transition to renewable energy sources.
"""

doc2_similar = """
Global warming is among the biggest threats to human civilization in modern times. 
Increasing temperatures worldwide are leading to melting polar ice, elevated ocean levels, 
and more erratic climate conditions. Researchers caution that failing to take urgent steps 
to decrease greenhouse gas emissions could result in disastrous outcomes such as severe 
water shortages, catastrophic flooding, and widespread loss of animal species. Nations and 
institutions need to collaborate on developing eco-friendly strategies and shifting toward 
clean energy alternatives.
"""

# ============================================================================
# SET 3: MODERATE SIMILARITY (60-75% Similar - Same topic, different focus)
# ============================================================================

doc3_healthcare_tech = """
Healthcare technology is transforming patient care and medical treatments. Electronic health 
records enable doctors to access patient information instantly, improving diagnosis accuracy 
and treatment efficiency. Telemedicine platforms allow patients to consult with specialists 
remotely, making healthcare more accessible to rural communities. Wearable devices monitor 
vital signs in real-time, alerting medical professionals to potential health issues before 
they become critical. Artificial intelligence assists radiologists in detecting diseases 
from medical imaging with unprecedented accuracy.
"""

doc3_medical_innovation = """
Modern medicine has witnessed extraordinary advancements in recent decades. Robotic surgery 
systems provide surgeons with enhanced precision and control during complex procedures. 
Gene therapy offers hope for treating previously incurable genetic disorders. Immunotherapy 
has revolutionized cancer treatment by harnessing the body's immune system to fight tumors. 
Personalized medicine tailors treatments to individual patients based on their genetic 
profiles, improving outcomes and reducing side effects. These innovations are extending 
lifespans and improving quality of life for millions of patients worldwide.
"""

# ============================================================================
# SET 4: LOW SIMILARITY (30-50% Similar - Related field, different content)
# ============================================================================

doc4_space_exploration = """
Space exploration continues to captivate human imagination and drive technological innovation. 
NASA's Mars rovers have discovered evidence of ancient water on the red planet, raising 
exciting possibilities about past life. Private companies like SpaceX are developing reusable 
rockets that dramatically reduce the cost of space travel. The James Webb Space Telescope is 
revealing unprecedented details about the early universe and distant galaxies. International 
collaboration on the International Space Station demonstrates humanity's ability to work 
together on ambitious scientific endeavors.
"""

doc4_ocean_research = """
Ocean exploration remains one of Earth's final frontiers. Scientists estimate that we have 
explored less than five percent of the world's oceans. Deep-sea submersibles are discovering 
bizarre creatures that thrive in extreme conditions near hydrothermal vents. Coral reefs, 
often called the rainforests of the sea, support incredible biodiversity but face threats 
from warming waters and pollution. Marine biology research is uncovering compounds from 
sea organisms that show promise for developing new medicines. Protecting ocean ecosystems 
is crucial for maintaining the health of our entire planet.
"""

# ============================================================================
# SET 5: PLAGIARISM TEST SET (Academic Context)
# ============================================================================

doc5_student_original = """
The French Revolution, which began in 1789, fundamentally transformed European political 
and social structures. The revolution emerged from widespread dissatisfaction with the 
absolute monarchy, economic inequality, and the privileges enjoyed by the nobility and 
clergy. The storming of the Bastille on July 14, 1789, symbolized the people's uprising 
against royal authority. Revolutionary leaders like Maximilien Robespierre advocated for 
radical changes including the abolition of feudalism and the establishment of a republic. 
The revolution's ideals of liberty, equality, and fraternity influenced democratic movements 
worldwide for centuries to come.
"""

doc5_student_plagiarized = """
The French Revolution started in 1789 and fundamentally transformed European political 
and social structures. This revolution emerged from widespread dissatisfaction with the 
absolute monarchy, economic inequality, and the privileges enjoyed by nobility and clergy. 
When the Bastille was stormed on July 14, 1789, it symbolized the uprising of people 
against royal authority. Revolutionary leaders such as Maximilien Robespierre advocated 
for radical changes including abolition of feudalism and establishment of a republic. 
Revolutionary ideals of liberty, equality, and fraternity influenced democratic movements 
worldwide for centuries.
"""

doc5_student_properly_paraphrased = """
Beginning in 1789, France experienced a revolutionary period that dramatically altered 
European governance and social systems. This upheaval stemmed from public frustration 
with monarchical absolutism, wealth disparity, and aristocratic advantages. The symbolic 
attack on the Bastille fortress represented popular resistance to monarchical power. 
Political figures like Robespierre championed sweeping reforms, including ending feudal 
systems and creating republican governance. These revolutionary principles subsequently 
inspired liberation movements across the globe throughout the following two centuries.
"""

# ============================================================================
# SET 6: TECHNOLOGY ARTICLES (Different Sub-topics)
# ============================================================================

doc6_cybersecurity = """
Cybersecurity threats are evolving rapidly in our increasingly connected world. Ransomware 
attacks have become more sophisticated, targeting not only businesses but also critical 
infrastructure like hospitals and power grids. Hackers employ social engineering tactics 
to trick employees into revealing sensitive information or installing malicious software. 
Organizations must implement multi-layered security strategies including firewalls, 
encryption, intrusion detection systems, and regular security audits. Employee training 
is crucial as human error remains one of the weakest links in cybersecurity defenses.
"""

doc6_blockchain = """
Blockchain technology extends far beyond cryptocurrency applications. This distributed 
ledger system provides transparent, tamper-proof record-keeping that can revolutionize 
supply chain management, voting systems, and digital identity verification. Smart contracts 
automatically execute agreements when predetermined conditions are met, eliminating the 
need for intermediaries and reducing transaction costs. Major corporations are exploring 
blockchain implementations to improve efficiency, reduce fraud, and increase transparency 
in their operations. However, scalability and energy consumption remain significant 
challenges that developers must address.
"""

doc6_quantum_computing = """
Quantum computing represents a paradigm shift in computational power and problem-solving 
capabilities. Unlike classical computers that use bits representing zeros and ones, quantum 
computers use qubits that can exist in multiple states simultaneously through superposition. 
This allows quantum computers to process complex calculations exponentially faster than 
traditional systems. Potential applications include drug discovery, financial modeling, 
climate prediction, and cryptography. Tech giants and research institutions are racing to 
achieve quantum supremacy and develop practical quantum computers that can solve real-world 
problems.
"""

# ============================================================================
# SET 7: BUSINESS & ECONOMICS
# ============================================================================

doc7_remote_work = """
Remote work has transformed from a rare perk to a standard employment option for millions 
of professionals. The pandemic accelerated this shift, forcing companies to rapidly adopt 
digital collaboration tools and flexible work policies. Many organizations have discovered 
that remote work increases employee satisfaction and productivity while reducing overhead 
costs. However, challenges include maintaining company culture, ensuring effective 
communication, and addressing employee isolation. Hybrid work models that combine office 
and remote work are emerging as a popular compromise that balances flexibility with 
in-person collaboration.
"""

doc7_ecommerce = """
Electronic commerce has revolutionized retail and consumer behavior. Online marketplaces 
enable businesses to reach global customers without physical storefronts. Mobile shopping 
apps provide convenient purchasing experiences with personalized recommendations based on 
browsing history and preferences. Social media platforms have integrated shopping features, 
blurring the lines between social networking and e-commerce. Supply chain innovations and 
sophisticated logistics networks enable rapid delivery, with same-day shipping becoming 
increasingly common in urban areas. Traditional retailers must adapt or face obsolescence 
in this digital marketplace.
"""

doc7_startup_culture = """
Startup culture emphasizes innovation, agility, and rapid growth over traditional corporate 
structures. These young companies often operate with flat hierarchies, encouraging all team 
members to contribute ideas and take ownership of projects. Risk-taking is celebrated, and 
failure is viewed as a learning opportunity rather than a career setback. Venture capital 
funding enables startups to scale quickly, though most fail to achieve profitability or 
sustainable growth. The startup ecosystem thrives in technology hubs like Silicon Valley, 
attracting entrepreneurial talent from around the world seeking to build the next 
breakthrough company.
"""

# ============================================================================
# SET 8: ENVIRONMENTAL SCIENCE
# ============================================================================

doc8_renewable_energy = """
Renewable energy sources are becoming increasingly cost-competitive with fossil fuels. 
Solar panel efficiency has improved dramatically while manufacturing costs have plummeted, 
making solar power accessible to homeowners and businesses worldwide. Wind energy farms, 
both onshore and offshore, generate clean electricity at scale. Battery storage technology 
advances are solving the intermittency problem associated with renewable sources. Governments 
are implementing incentives and regulations to accelerate the transition to clean energy. 
However, grid infrastructure must be upgraded to handle distributed renewable generation 
and maintain reliability.
"""

doc8_biodiversity_loss = """
Biodiversity loss represents an ecological crisis with far-reaching consequences for 
ecosystems and human welfare. Habitat destruction, primarily driven by agricultural 
expansion and urbanization, threatens countless species with extinction. Deforestation 
eliminates crucial wildlife habitats while also reducing the planet's capacity to absorb 
carbon dioxide. Overfishing has depleted ocean fish populations, disrupting marine food 
chains. Invasive species introduced by human activity outcompete native organisms, 
fundamentally altering ecosystems. Conservation efforts including protected areas, 
wildlife corridors, and species reintroduction programs are essential for preserving 
Earth's biological diversity.
"""

doc8_sustainable_agriculture = """
Sustainable agriculture practices aim to produce food while minimizing environmental 
impact and preserving resources for future generations. Crop rotation maintains soil 
health and reduces pest problems without excessive pesticide use. Precision farming 
techniques use GPS technology and data analytics to optimize irrigation, fertilization, 
and harvesting, reducing waste and improving yields. Organic farming eliminates synthetic 
chemicals, promoting soil biodiversity and producing healthier food. Vertical farming in 
urban areas maximizes space efficiency and reduces transportation costs. These approaches 
demonstrate that agricultural productivity and environmental stewardship can coexist.
"""

# ============================================================================
# SET 9: EDUCATION & LEARNING
# ============================================================================

doc9_online_learning = """
Online education has democratized access to knowledge and skills training. Massive Open 
Online Courses (MOOCs) from prestigious universities reach millions of learners worldwide, 
breaking down geographical and financial barriers to education. Interactive video lessons, 
virtual labs, and peer collaboration tools create engaging learning experiences. Adaptive 
learning algorithms personalize content difficulty and pacing to individual student needs. 
However, online learning requires strong self-discipline and time management skills. 
Concerns about academic integrity, student engagement, and the lack of hands-on experience 
in certain fields remain ongoing challenges.
"""

doc9_gamification = """
Gamification applies game design principles to educational contexts to increase student 
motivation and engagement. Point systems, badges, leaderboards, and achievement levels 
tap into competitive instincts and provide immediate feedback. Quest-based learning 
structures complex topics into manageable challenges with clear objectives. Narrative 
elements create context and meaning for educational content, making abstract concepts 
more relatable. Research shows gamification can improve learning outcomes, particularly 
for repetitive practice and skill development. However, poorly designed gamification can 
distract from learning objectives or create unhealthy competition among students.
"""

# ============================================================================
# SET 10: SPORTS & FITNESS
# ============================================================================

doc10_sports_analytics = """
Sports analytics has revolutionized how teams evaluate players, develop strategies, and 
make decisions. Advanced statistics go beyond traditional metrics to measure player 
efficiency, court or field positioning, and situational performance. Video analysis 
software breaks down player movements frame by frame, identifying areas for technique 
improvement. Wearable sensors track athlete biometrics during training and competition, 
helping optimize performance and prevent injuries. Teams employ data scientists and 
statistical analysts to gain competitive advantages. However, some argue that excessive 
reliance on analytics overlooks intangible qualities like leadership and team chemistry.
"""

doc10_fitness_trends = """
Fitness trends continuously evolve, reflecting changing preferences and scientific 
understanding of exercise physiology. High-intensity interval training (HIIT) has gained 
popularity for its efficiency in burning calories and improving cardiovascular fitness in 
short workout sessions. Functional fitness emphasizes movements that translate to daily 
activities rather than isolated muscle training. Mind-body practices like yoga and Pilates 
combine physical exercise with mental wellness benefits. Boutique fitness studios offer 
specialized, community-focused workout experiences. Fitness technology including smart 
watches and connected equipment provides real-time performance feedback and motivation.
"""

# ============================================================================
# WRITE ALL DOCUMENTS TO FILES
# ============================================================================

documents = {
    # Exact Duplicates (100%)
    '01_AI_Original.txt': doc1_original,
    '02_AI_Duplicate.txt': doc1_duplicate,
    
    # High Similarity (85-95%)
    '03_Climate_Original.txt': doc2_original,
    '04_Climate_Similar.txt': doc2_similar,
    
    # Moderate Similarity (60-75%)
    '05_Healthcare_Tech.txt': doc3_healthcare_tech,
    '06_Medical_Innovation.txt': doc3_medical_innovation,
    
    # Low Similarity (30-50%)
    '07_Space_Exploration.txt': doc4_space_exploration,
    '08_Ocean_Research.txt': doc4_ocean_research,
    
    # Plagiarism Test Set
    '09_Student_Original.txt': doc5_student_original,
    '10_Student_Plagiarized.txt': doc5_student_plagiarized,
    '11_Student_Paraphrased.txt': doc5_student_properly_paraphrased,
    
    # Technology Articles
    '12_Cybersecurity.txt': doc6_cybersecurity,
    '13_Blockchain.txt': doc6_blockchain,
    '14_Quantum_Computing.txt': doc6_quantum_computing,
    
    # Business & Economics
    '15_Remote_Work.txt': doc7_remote_work,
    '16_Ecommerce.txt': doc7_ecommerce,
    '17_Startup_Culture.txt': doc7_startup_culture,
    
    # Environmental Science
    '18_Renewable_Energy.txt': doc8_renewable_energy,
    '19_Biodiversity_Loss.txt': doc8_biodiversity_loss,
    '20_Sustainable_Agriculture.txt': doc8_sustainable_agriculture,
    
    # Education & Learning
    '21_Online_Learning.txt': doc9_online_learning,
    '22_Gamification.txt': doc9_gamification,
    
    # Sports & Fitness
    '23_Sports_Analytics.txt': doc10_sports_analytics,
    '24_Fitness_Trends.txt': doc10_fitness_trends,
}

# Write all documents
for filename, content in documents.items():
    with open(f'test_documents/{filename}', 'w', encoding='utf-8') as f:
        f.write(content.strip())

print(f"✅ Successfully created {len(documents)} test documents in 'test_documents' folder!")
print("\n📊 Test Document Categories:")
print("=" * 60)
print("1. Exact Duplicates (100% similarity): 2 documents")
print("2. High Similarity (85-95%): 2 documents")
print("3. Moderate Similarity (60-75%): 2 documents")
print("4. Low Similarity (30-50%): 2 documents")
print("5. Plagiarism Test Set: 3 documents")
print("6. Technology Articles: 3 documents")
print("7. Business & Economics: 3 documents")
print("8. Environmental Science: 3 documents")
print("9. Education & Learning: 2 documents")
print("10. Sports & Fitness: 2 documents")
print("=" * 60)
print(f"\n📁 Total: {len(documents)} test documents created")
print("\n💡 Use these documents to test:")
print("   - Exact duplicate detection")
print("   - High similarity plagiarism")
print("   - Topic-based clustering (should create ~6-8 clusters)")
print("   - Similarity threshold adjustments")
print("   - Network visualization with various connection levels")