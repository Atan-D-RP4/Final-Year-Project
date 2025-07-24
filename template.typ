#let document-title = "Opinion Mining on Product Reviews"
#let submission-details = [
  A Project report submitted in partial fulfillment of the requirements for the award of degree in        
  *BACHELOR OF TECHNOLOGY* \
  *(COMPUTER SCIENCE AND ENGINEERING)*

  *SUBMITTED BY* \
  #table(
    columns: (auto, auto),
    align: left,
    [Registration number], [Name of the Student],
    [A22126510164], [M. Veerendra Kumar],
    [A22126510139], [CH. Kavya Naidu],
    [A22126510174], [P. Prasanth],
    [A22126510178], [S. Sai Praveen],
    [A22126510145], [Gadde Vivek],
  )

  *UNDER THE GUIDANCE OF* \
  Y. Sujatha \                                                                                              Associate professor

  #box(width: 2.4in, height: 2.4in)[
    #image("anits_logo.png", width: 2.4in, height: 2.4in)
  ]

  *DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING* \
  *ANIL NEERUKONDA INSTITUTE OF TECHNOLOGY AND SCIENCES (A+)* \
  SANGIVALASA, VISAKHAPATNAM – 531162 \
  July - 2025
]

#set document(title: document-title)
#set page(
  margin: (x: 1in, y: 1in),
  numbering: "i",
)
#set text(font: "Times New Roman", size: 12pt)
#set heading(numbering: "1.")
#set par(justify: true, leading: 1em)

#align(center)[
  #text(size: 16pt, weight: "bold")[#document-title]
  #v(1em)
  #submission-details
]

#pagebreak()

#set page(numbering: "1")

= CONTEXT

- About Domain
- How it is feasible to present society needs
- What the problem identified
- What the solution suggested
- How the proposed solution apt for present needs of the users
- Technologies used
- Feasibility study of your application
  - Operational feasibility
  - Technical feasibility
  - Economical feasibility
- Any architecture model for your proposed system
- References (if any existing papers)
- Conclusion

= About Domain

The domain of this project lies at the intersection of *Decentralized Online Social Networks (DOSNs)*, *Privacy-Preserving Computing*, *Cloud Security*, and *Trusted Computing*. As online interactions increasingly move to digital platforms, social networks have become central to communication, collaboration, and information sharing. However, the centralized nature of traditional Online Social Networks (OSNs) such as Facebook, Twitter, and Instagram poses significant challenges in terms of user data privacy, control, and trust.

In centralized OSNs, all user data is stored on servers controlled by the service provider. This creates single points of failure and control, allowing providers to access, exploit, or leak user data, either intentionally or through security breaches. Additionally, centralized providers often collect metadata, track user behavior, and monetize personal information, leading to privacy violations and loss of data ownership.

To address these issues, the domain has shifted towards decentralized models such as *Vis-à-Vis*, which introduces the concept of *Virtual Individual Servers (VISs)*. In Vis-à-Vis, each user’s data is hosted on a dedicated virtual server within a cloud infrastructure, giving them greater control over access permissions and data sharing. This model removes the reliance on a central authority, thus improving privacy, transparency, and data sovereignty.

However, this decentralization introduces new challenges:
- Cloud providers can still access unencrypted data.
- VIS integrity is hard to verify without widespread adoption of *Trusted Platform Modules (TPMs)*.
- Malware and tampering risks threaten the authenticity of VIS behavior.

Hence, the domain evolves to incorporate cutting-edge techniques for privacy protection and threat mitigation. This includes:
- *Machine Learning (ML)*: Used for real-time detection of suspicious behavior, anomalies, or unauthorized access patterns within the VIS.
- *Fully Homomorphic Encryption (FHE)*: Enables computation on encrypted data without decryption, ensuring that even if cloud providers access data, they cannot understand or misuse it.

This domain not only enhances technical privacy but also supports user-centric design, where individuals regain control over their personal information. It holds vast potential in shaping secure alternatives to conventional social networks, especially in sensitive applications like healthcare, political discourse, education, and activist communities where data misuse can have severe consequences.

Ultimately, this domain is foundational in building trustworthy, scalable, and privacy-aware digital ecosystems, a key demand in today's interconnected and surveillance-prone digital world.

= How It Is Feasible to Present Society Needs?

In today’s digital age, privacy and data protection have become critical societal needs. As people increasingly rely on online social networks (OSNs) for communication, professional networking, activism, and content sharing, the control over personal data has become a pressing issue. Traditional centralized platforms collect, store, and process user data, often violating user trust through data monetization, breaches, and opaque algorithms.

This project addresses these concerns by offering a decentralized and privacy-preserving framework that empowers users with ownership and autonomy over their data. Here’s how it aligns with and fulfills present societal needs:

1. *User Empowerment and Data Ownership* \
  Society demands platforms where individuals are no longer passive data sources. By deploying *Virtual Individual Servers (VISs)*, users host and manage their own data in isolated environments, minimizing reliance on third-party platforms. This satisfies the public’s increasing desire for control, transparency, and informed consent over their digital presence.

2. *Protection Against Surveillance and Breaches* \
  Recent incidents of mass surveillance, corporate espionage, and social media data leaks (e.g., Cambridge Analytica scandal) have made people wary of centralized platforms. This project’s integration of *Fully Homomorphic Encryption (FHE)* ensures that even when data is stored in cloud servers, it remains unintelligible to outsiders, addressing societal demand for robust privacy.

3. *Trustworthy Communication Infrastructure* \
  Trust in platforms is eroding. By incorporating *Machine Learning (ML)* techniques for real-time anomaly detection and tampering identification, the system proactively defends against malware, unauthorized changes, and insider threats. This aligns with society’s need for safe, accountable digital spaces.

4. *Decentralization and Resilience* \
  During political crises or internet shutdowns, centralized platforms can be censored or disabled. A decentralized system like Vis-à-Vis offers resilience and freedom from centralized authority, which appeals strongly to civil society groups, journalists, activists, and marginalized communities who face censorship or surveillance.

5. *Ethical and Transparent Technology* \
  There is a growing societal demand for ethical AI and responsible technology. By combining cryptographic privacy (FHE) with intelligent threat detection (ML), this project ensures privacy isn’t compromised for utility. It sets a new ethical standard for how social networks can operate with respect to user rights.

6. *Adaptability Across Demographics* \
  The solution is applicable not only to tech-savvy users but also to general users through user-friendly interfaces and integration with existing platforms (e.g., Facebook via browser extensions). This increases its feasibility and accessibility for widespread adoption.

= What the Problem Identified

Despite the emergence of decentralized frameworks like *Vis-à-Vis*, which use *Virtual Individual Servers (VISs)* to give users control over their data, critical privacy and security challenges persist. The core problems identified are:

1. *Exposure of Data to Cloud Providers* \
  VISs are hosted on third-party cloud utilities (e.g., Amazon EC2). Although decentralized in architecture, these cloud providers still have access to unencrypted user data, meaning they can potentially read, misuse, or leak sensitive personal information. This contradicts the goal of complete user privacy.

2. *Lack of Software Integrity Verification* \
  Currently, there is no universal enforcement of *Trusted Platform Modules (TPMs)* across cloud platforms. TPMs are essential for ensuring that the VIS is running genuine, untampered software. Without this, users cannot verify whether their VIS has been compromised, leaving them vulnerable to:
  - Malware injection
  - Data manipulation
  - Unauthorized surveillance

3. *Inability to Detect Real-time Tampering or Abnormal Behavior* \
  Traditional VIS systems lack intelligent monitoring. If an attacker alters the VIS behavior, such as leaking data or modifying permissions, the system cannot identify or react to it in real-time. This creates a silent failure risk, especially dangerous in sensitive environments.

4. *False Assumption of Trust in Infrastructure* \
  Vis-à-Vis assumes that users trust compute utilities (cloud providers) not to abuse their access. In reality, this trust assumption is weak, especially when regulations, business models, or malicious insiders can easily violate this trust. Hence, infrastructure remains a vulnerable component.

5. *Lack of Secure Computation Mechanisms* \
  Even when VISs store private data securely, processing that data still requires decryption, exposing it during use. There is no mechanism in traditional Vis-à-Vis to allow for computations on encrypted data, which is critical for analytics, personalization, and other services.

= What the Solution Suggested

To address the vulnerabilities in existing decentralized social networks like Vis-à-Vis, this project proposes a hybrid, intelligent privacy-preserving framework that combines two powerful technologies:

1. *Integration of Machine Learning (ML) for Real-time Anomaly Detection* \
  The system incorporates ML models within each user's VIS to continuously monitor for:
  - Unusual access patterns
  - Behavioral deviations
  - Signs of tampering or unauthorized control \
  These models are trained to detect both internal (malware, insider threats) and external (unauthorized user access, privilege escalation) anomalies. Once an anomaly is detected, the system can automatically:
  - Trigger alerts to the user
  - Revoke access
  - Freeze suspicious operations
  - Log and report the event for forensic analysis \
  This proactive defense layer ensures real-time response to threats, reducing the risk of unnoticed data leakage or manipulation.

2. *Use of Fully Homomorphic Encryption (FHE) for Privacy-preserving Computation* \
  To solve the problem of cloud provider access to sensitive data, the framework uses *Fully Homomorphic Encryption*, which allows computations to be performed directly on encrypted data, without needing to decrypt it. \
  With FHE:
  - VISs can perform analytics, search, and filtering operations on user data without ever revealing the plaintext.
  - Cloud providers can manage and compute over data without having any visibility into its content.
  - Users retain mathematical guarantees of confidentiality, even during active data processing. \
  This technique eliminates trust dependencies on the infrastructure by ensuring that data remains private at all stages—at rest, in transit, and in use.

3. *Software Integrity with TPM-aware Compatibility* \
  Although the current infrastructure lacks universal TPM support, this system is designed to be TPM-compatible, so that as TPM adoption grows:
  - VIS software stacks can attest their integrity.
  - Nodes can verify each other before participating in peer operations.
  - Users can be confident that their VIS is running genuine, unmodified software. \
  This enables future-proofing of the solution against deeper integrity threats.

4. *Seamless Integration with Existing OSNs* \
  Instead of building a new social network from scratch, the solution allows interoperability with existing platforms (e.g., Facebook) via:
  - Browser extensions
  - APIs (like Facebook Connect)
  - Embedded links to the user's VIS \
  This ensures that users can retain their social graph while shifting data control to their own secure, intelligent VIS environments.

*Summary of the Proposed Solution:*

#table(
  columns: (auto, auto),
  align: left,
  [*Challenge*], [*Solution Component*],
  [Cloud access to data], [Fully Homomorphic Encryption (FHE)],
  [No software integrity guarantee], [TPM support + attestation (optional)],
  [Undetected tampering/malware], [ML-based anomaly and behavior detection],
  [No privacy during computation], [FHE-secured computation],
  [Integration barriers with OSNs], [APIs and extensions for easy adoption],
)