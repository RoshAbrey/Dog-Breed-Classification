import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import base64
import pandas as pd

class_names = {
        0: 'Chihuahua',
        1: 'Japanese Spaniel',
        2: 'Maltese Dog',
        3: 'Pekinese',
        4: 'Shih-Tzu',
        5: 'Blenheim Spaniel',
        6: 'Papillon',
        7: 'Toy Terrier',
        8: 'Rhodesian Ridgeback',
        9: 'Afghan Hound',
        10: 'Basset',
        11: 'Beagle',
        12: 'Bloodhound',
        13: 'Bluetick',
        14: 'Black and Tan Coonhound',
        15: 'Walker Hound',
        16: 'English Foxhound',
        17: 'Redbone',
        18: 'Borzoi',
        19: 'Irish Wolfhound',
        20: 'Italian Greyhound',
        21: 'Whippet',
        22: 'Ibizan Hound',
        23: 'Norweigian Elkhound',
        24: 'Otterhound',
        25: 'Saluki',
        26: 'Scottish Deerhound',
        27: 'Weimaraner',
        28: 'Staffordshire Bullterrier',
        29: 'American Staffordshire Bullterrier',
        30: 'Bedlington Terrier',
        31: 'Border Terrier',
        32: 'Kerry Blue Terrier',
        33: 'Irish Terrier',
        34: 'Norflok Terrier',
        35: 'Norwich Terrier',
        36: 'Yorkshire terrier',
        37: 'Wire haired fox terrier',
        38: 'Lakeland Terrier',
        39: 'Sealyham Terrier',
        40: 'Airedale',
        41: 'Cairn',
        42: 'Australian Terrier',
        43: 'Dandie Dinmont',
        44: 'Boston bull',
        45: 'Miniature Schnauzer',
        46: 'Giant Schnauzer',
        47: 'Standard Schnauzer',
        48: 'Scotch Terrier',
        49: 'Tibetan Terrier',
        50: 'Silky Terrier',
        51: 'Soft Coated Wheaten Terrier',
        52: 'West Highland White Terrier',
        53: 'Lhasa',
        54: 'Flat Coated Retriever',
        55: 'Curly Coated Retriever',
        56: 'Golden retriever', 
        57: 'Labrador retriever',
        58: 'Chesapeake bay retriever',
        59: 'German short haired pointer',
        60: 'Vizsla',
        61: 'English Setter',
        62: 'Irish Setter',
        63: 'Gordon Setter',
        64: 'Brittany Spaniel',
        65: 'Clumber',
        66: 'English Springer',
        67: 'Welsh springer spaniel',
        68: 'Cocker Spaniel',
        69: 'Sussex Spaniel',
        70: 'Irish Water Spaniel',
        71: 'Kuvasz',
        72: 'Schipperke',
        73: 'Groenendael',
        74: 'Malinois',
        75: 'Briard',
        76: 'Kelpie',
        77: 'Komondor',
        78: 'Old English Sheepdog',
        79: 'Shetland Sheepdog',
        80: 'Collie',
        81:  'Border Collie',
        82: 'Bouvier Des Flandres',
        83: 'Rottweiler',
        84: 'German Shepherd',
        85: 'Doberman',
        86: 'Miniature Pinscher',
        87: 'Greater Swiss Mountain Dog',
        88: 'Bernese Mountain Dog',
        89: 'Appenzeller',
        90: 'EntleBucher',
        91:  'Boxer',
        92: 'Bull Mastiff',
        93: 'Tibetan MAstiff',
        94: 'French Bulldog',
        95: 'Great Dane',
        96: 'Saint Bernard',
        97: 'Eskimo Dog',
        98: 'Malamute',
        99: 'Siberian Husky',
        100: 'Affenpinscher',
        101:  'Basenji',
        102: 'Pug',
        103: 'Leonberg',
        104: 'Newfoundland',
        105: 'Great Pyreness',
        106: 'Samoyed',
        107: 'Pomeranian',
        108: 'Chow',
        109: 'Keeshond',
        110: 'Brabancon Griffon',
        111:  'Pembroke',
        112: 'Cardigan',
        113: 'Toy Poodle',
        114: 'Miniature Poodle',
        115: 'Standard Poodle',
        116: 'Mexican Hairless',
        117: 'Dingo',
        118: 'Dhole',
        119: 'African Hunting Dog'
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('C:/Users/My PC/Desktop/Computer Science/TESTING-MODEL-Enhanced.h5')
    return model
def load_model_base():
    model = tf.keras.models.load_model('C:/Users/My PC/Desktop/Computer Science/TESTING-MODEL.h5') #base model to be inserted here <<-
    return model

model = load_model()
model_base = load_model_base()
def import_and_predict_proposed(uploaded_image, model):
    size = (224, 224)
    
    # Convert the image to RGB mode
    image = uploaded_image.convert("RGB")
    
    # Resize and preprocess the image
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.array(image)  # Convert PIL image to NumPy array
    image_array = image_array / 255.0  # Normalize pixel values
    
    # Expand dimensions to match the model's input shape
    image_tensor = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    prediction = model.predict(image_tensor)[0]
    # Get the indices of the top 3 predicted classes
    top_3_indices = np.argsort(prediction)[::-1][:3]
    
    # Prepare table data
    table_data = []
    for idx in top_3_indices:
        breed_label = class_names[idx]  
        probability = prediction[idx]
        table_data.append([breed_label, f"{probability * 100:.2f}%"])
    
    # Convert table data to DataFrame
    table_df = pd.DataFrame(table_data, columns=["Breed", "Probability"])
    
    return table_df

def import_and_predict_baseline(uploaded_image, model):
    size = (224, 224)
    
    # Convert the image to RGB mode
    image = uploaded_image.convert("RGB")
    
    # Resize and preprocess the image
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.array(image)  # Convert PIL image to NumPy array
    image_array = image_array / 255.0  # Normalize pixel values
    
    # Expand dimensions to match the model's input shape
    image_tensor = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    prediction = model.predict(image_tensor)[0]
    
    # Get the indices of the top 3 predicted classes
    top_3_indices = np.argsort(prediction)[::-1][:3]
    
    # Prepare table data
    table_data = []
    for idx in top_3_indices:
        breed_label = class_names[idx]  # Assuming 'class_names' contains the label names
        probability = prediction[idx]
        table_data.append([breed_label, f"{probability * 100:.2f}%"])
    
    # Convert table data to DataFrame
    table_df = pd.DataFrame(table_data, columns=["Breed", "Probability"])
    
    return table_df


def run():
    img1 = Image.open('images.jpg')
    img1 = img1.resize((700, 350))
    st.image(img1, use_column_width=False)

    st.markdown(
        """
        <h1 style="text-align: center;">DOG BREED CLASSIFICATION SIMULATOR</h1>
        <h4 style="text-align: center; color: #d73b5c;">The trained data consists of a collection of labeled images of 120 different dog breeds.</h4>
        """,
        unsafe_allow_html=True
    )

    st.markdown('---')

    st.markdown(
        """
        <h3 style="text-align: center;">Upload an Image</h3>
        <p style="text-align: center;">Please upload an image of a dog to analyze its breed.</p>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.text("Please upload an image file!")
    else:
        img = Image.open(io.BytesIO(uploaded_file.read()))


        img_str = img_to_base64(img)
        st.markdown(
            f'<div style="text-align: center;"><img src="data:image/png;base64,{img_str}" alt="Uploaded Image" width="400px"></div>',
            unsafe_allow_html=True
        )
        st.markdown('---')

        st.success('Image uploaded successfully!')

        table_data = import_and_predict_proposed(img, model)
        table_data_base = import_and_predict_baseline(img, model_base)
        # Display the table of predicted breeds and probabilities
        
        st.markdown('---')
        
        # Display the top probability output breed in the specified format
        col1, col2 = st.columns(2)
        with col1:  
            breed_label = table_data["Breed"].iloc[0]
            st.markdown("<h2 style='text-align: center;'><span style='color: orange;'>Predicted Breed (Enhanced):  </span><span style='color: green;'>{}</span></h2>".format(breed_label), unsafe_allow_html=True)
            st.table(table_data)
        with col2:
            breed_label = table_data_base["Breed"].iloc[0]
            st.markdown("<h2 style='text-align: center;'><span style='color: orange;'>Predicted Breed (Baseline):  </span><span style='color: green;'>{}</span></h2>".format(breed_label), unsafe_allow_html=True)
            st.table(table_data_base)
        st.markdown('---')
        
        #chihuahua
        if breed_label == "Chihuahua":
            tab1, tab2, tab3= st.tabs(["Cleft palate", "Cryptorchidism", "Keratitis sicca"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cleft Palate") 
                    st.image("https://www.bpmcdn.com/f/files/victoria/import/2021-06/25618240_web1_210605-CPL-SPCA-Lock-In-Fore-Love-Baby-Snoot-Chilliwack_4.jpg", width=200)
                with col2:
                    Cleft_Palate = '''A condition where the roof of the mouth is not closed 
                                and the inside of the nose opens into the mouth. It occurs due to a failure of the roof of the mouth to close during 
                                development in the womb. This results in a hole between the mouth and the nasal cavity. 
                                The defect can occur in the lip (primary cleft palate) or along the roof of the mouth (secondary cleft palate).
                            '''
                    st.markdown(Cleft_Palate)
                with st.expander("See More details"):
                    st.subheader("Cleft palate in puppies Prognosis")
                    st.write("A cleft palate is generally detected by visual examination of newborn puppies by the veterinary surgeon or breeder. Cleft palate of the lip or hard palate are easy to see, but soft palate defects can sometimes require sedation or general anaesthesia to visualise. Affected puppies will often have difficulty suckling and swallowing. This is often seen as coughing, gagging, and milk bubbling from the pup’s nose. In less severe defects, more subtle signs such as sneezing, snorting, failure to grow, or sudden onset of breathing difficulty (due to aspiration of milk or food) can occur.")
                    st.markdown("---")
                    st.subheader("Treatment for cleft palate in puppies")
                    st.write("Treatment depends on the severity of the condition, the age at which the diagnosis is made, and whether there are complicating factors, such as aspiration pneumonia.")
                    st.write("Small primary clefts of the lip and nostril of the dog are unlikely to cause clinical problems.")
                    st.write("Secondary cleft palates in dogs require surgical treatment to prevent long-term nasal and lung infections and to help the puppy to feed effectively. The surgery involves either creating a single flap of healthy tissue and overlapping it over the defect or creating a ‘double flap’, releasing the palate from the inside of the upper teeth, and sliding it to meet in the middle over the defect.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cryptorchidism")
                    st.image("https://iloveveterinary.com/wp-content/uploads/2021/05/Cryptorchid-Chihuahua.jpg.webp", width=200)
                with col2:
                    Cryptorchidism = '''the medical term that refers to the failure of one or both testicles (testes) to descend into the scrotum. The testes develop near the kidneys within the abdomen and normally descend into the scrotum by two months of age. In certain dogs, it may occur later, but rarely after six months of age. Cryptorchidism may be presumed to be present if the testicles cannot be felt in the scrotum after two to four months of age.
                            '''
                    st.markdown(Cryptorchidism)
                with st.expander("See More details"):
                    st.subheader("If the testicles aren't in the scrotum, where are they?")
                    st.write("In most cases of cryptorchidism, the testicle is retained in the abdomen or in the inguinal canal (the passage through the abdominal wall into the genital region through which a testicle normally descends). Sometimes, the testicle will be located in the subcutaneous tissues (just under the skin) in the groin region, between the inguinal canal and the scrotum.")
                    st.markdown("---")
                    st.subheader("How is cryptorchidism diagnosed?")
                    st.write("In cases of abdominal cryptorchidism, the testicle cannot be felt from the outside. An abdominal ultrasound or radiographs (X-rays) may be performed to determine the exact location of the retained testicle, but this is not often done before surgery, as it is not required to proceed with surgery. Typically, only one testicle is retained, and this is called unilateral cryptorchidism. If you have a dog that does not appear to have testicles but is exhibiting male behaviors, a hormonal test called an hCG stimulation test can be performed to see if he is already neutered.")
                    st.markdown("---")
                    st.subheader("What causes cryptorchidism and how common is it?")
                    st.write("Cryptorchidism occurs in all breeds but toy breeds, including toy Poodles, Pomeranians, and Yorkshire Terriers, may be at higher risk. Approximately 75% of cases of cryptorchidism involve only one retained testicle while the remaining 25% involve failure of both testicles to descend into the scrotum. The right testicle is more than twice as likely to be retained as the left testicle. Cryptorchidism affects approximately 1-3% of all dogs. The condition appears to be inherited since it is commonly seen in families of dogs, although the exact cause is not fully understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of cryptorchidism?")
                    st.write("This condition is rarely associated with pain or other signs unless a complication develops. In its early stages, a single retained testicle is significantly smaller than the other, normal testicle. If both testicles are retained, the dog may be infertile. The retained testicles continue to produce testosterone but generally fail to produce sperm.")
                    st.markdown("---")
                    st.subheader("What is the treatment for cryptorchidism?")
                    st.write("Neutering and removal of the retained testicle(s) are recommended. If only one testicle is retained, the dog will have two incisions - one for extraction of each testicle. If both testicles are in the inguinal canal, there will also be two incisions. If both testicles are in the abdomen, a single abdominal incision will allow access to both.")
                    st.markdown("---")
                    st.subheader("What if I don't want to neuter my dog?")
                    st.write("There are several good reasons for neutering a dog with cryptorchidism. The first reason is to remove the genetic defect from the breed line. Cryptorchid dogs should never be bred. Second, dogs with a retained testicle are more likely to develop a testicular tumor (cancer) in the retained testicle. Third, as described above, the testicle can twist, causing pain and requiring emergency surgery to correct. Finally, dogs with a retained testicle typically develop the undesirable characteristics associated with intact males like urine marking and aggression. The risk of developing testicular cancer is estimated to be at least ten times greater in dogs with cryptorchidism than in normal dogs.")
                    st.markdown("---")
                    st.subheader("What is the prognosis for a dog with cryptorchidism?")
                    st.write("The prognosis is excellent for dogs that undergo surgery early before problems develop in the retained testicle. The surgery is relatively routine, and the outcomes are overwhelmingly positive.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Keratitis sicca")
                    st.image("https://images.ctfassets.net/4dmg3l1sxd6g/6ofoq7dPDVcxWBc3WMmwwu/17a5b491521dec9c5558f2b51e88e3b0/dry-eye-in-dogs_figure-4-35281-article.png_-_en", width=200)
                with col2:
                    Keratitis_sicca = '''Dry eye syndrome in dogs, also known as Keratoconjunctivitis Sicca (KCS), involves decreased or inadequate tear production. Tears are important to the lubrication, comfort, and overall health of a dog’s eyes. '''
                    st.markdown(Keratitis_sicca)
                with st.expander("See More Details"):
                    st.subheader("What Is Dry Eye Syndrome in Dogs?")
                    st.write("Tears also contain antibacterial proteins, mucus, white blood cells to fight infection, and other enzymes to help keep the eyes clear and free of debris, infection, and irritations. ")
                    st.markdown("---")
                    st.subheader("Symptoms of Dry Eye Syndrome in Dogs")
                    st.write("Dogs with dry eye syndrome can exhibit one or many of the following symptoms: ")
                    st.write("Red, inflamed, irritated, and painful eyes ")
                    st.write("Redness and swelling of the conjunctiva, or the tissues around the eye ")
                    st.write("Frequent squinting and blinking ")
                    st.write("Dryness on the surface of the cornea—the outer clear dome-shaped part of the eye ")
                    st.write("Mucous-like discharge on the cornea (may be yellow or green if a secondary bacterial infection is present) ")
                    st.write("Obvious defects and irregularities of the cornea, including increased vascularization (abnormal growth of blood vessels to the injured area) and pigmentation as the eye attempts to heal and protect itself ")
                    st.write("Possible vision impairment and blindness ")
                    st.markdown("---")
                    st.subheader("Causes of Dry Eye Syndrome in Dogs")
                    st.write("The cause of dry eye syndrome in a dog can be due to one or a few underlying conditions. Your veterinarian will be able to determine what may have caused your pet’s diagnosis based on the dog’s medical history and an exam. Some of the underlying causes may be due to: ")
                    st.write("Immune system dysfunction: Most cases of dry eye syndrome in dogs are caused by the immune system attacking and destroying the lacrimal and third eyelid gland. Unfortunately, veterinarians do not know why this happens.  ")
                    st.write("Medications: Certain drugs can cause dry eye syndrome as a side effect, usually very shortly after a dog starts taking these medications. This type of dry eye syndrome can be temporary and may go away once the medication is discontinued. However, permanent damage can be done, and there is no way to predict which animals will have dry eye syndrome or how long it will last. Be sure to talk with your vet about possible side effects of all medications.  ")
                    st.write("Genes: Congenital alacrimia is a genetic form of dry eye syndrome and occurs in some breeds, most notably Yorkshire terriers. This is typically noticed in only one eye.  ")
                    st.write("Endocrine conditions: Some systemic disease (such as hypothyroidism, diabetes, and Cushing’s disease) frequently decrease tear production. ")
                    st.write("Infectious diseases: Canine distemper virus, leishmaniasis, and chronic blepharoconjunctivitis can all lead to dry eye syndrome. ")
                    st.write("Medical procedures: A common abnormality of dogs is a prolapsed third eyelid gland (more commonly known as cherry eye). While it is not recommended, some surgeons remove the gland entirely, leading to permanent decreased tear production. Local radiation for tumors can also cause permanent damage to the lacrimal and third eyelid glands. ")
                    st.write("Neurological problems: Loss of nerve function to the glands (commonly secondary to an inner ear infection) can decrease or stop production of tears. ")
                    st.write("Traumatic injury: Dry eye syndrome can occur with damage to the glands after severe inflammation or injury (such as from wounds or car accidents). ")
                    st.write("Transient causes: Anesthesia causes a temporary loss of tear production, as does the medication atropine. Once these are removed, tear production normally returns.  ")
                    st.markdown("---")
                    st.subheader("How Veterinarians Diagnose Dry Eye Syndrome in Dogs")
                    st.write("Vets use the Schirmer Tear Test (STT) to diagnose dry eye syndrome and measure aqueous tear production in dogs. This is a simple, painless test involving a strip of special paper placed in the lower eyelid. The moisture and tears from the eye wick onto the paper for 60 seconds. At the end of that time, the vet measures the tear production on the paper. For test results, more than 15 millimeters of tear production per minute is normal, while less than 10 millimeters indicates dry eye syndrome. Your vet may repeat the test to confirm the diagnosis. After the STT, your vet may also perform a fluorescein stain test to look for corneal ulcers. The stain makes an ulcer glow bright green under a black light. The vet may also use a test of intraocular pressure to look for inflammation or glaucoma. These conditions are common with dry eye and important to diagnose and treat at the same time.")
                    st.markdown("---")
                    st.subheader("Treatment of Dry Eye Syndrome in Dogs")
                    st.write("Lacrimostimulants: Most commonly, vets prescribe ophthalmic cyclosporine (a class of medications) or tacrolimus to stimulate tear production. Cyclosporine, when applied in the eye, keeps the immune system from harming the lacrimal and third eyelid glands, thus allowing tear restoration. Tacrolimus is typically used only if cyclosporine fails.  ")
                    st.write("Lacrimomimetics: Artificial tears moisten the surface of the eye, improve comfort, and help flush debris and allergens. These eye lubricants are essential to use with primary medications for dry eye syndrome, like cyclosporine, especially early in the treatment process when tear production hasn’t fully recovered. Only use artificial tears if your vet directs you. ")
                    st.write("Antibiotics: Bacterial infections and corneal ulcerations may also require broad-spectrum topical antibiotics. Dogs whose dry eye syndrome is related to the nervous system are treated with pilocarpine, which stimulates glandular secretion.  ")
                    st.write("Surgery: Dogs who don’t respond to treatment may require a surgery called parotid duct transposition, which carefully redirects saliva glands in the dog’s mouth to the eye, so that saliva can be used as tears. ")
                    st.markdown("---")
        #Japanese Spaniel
        elif breed_label == "Japanese Spaniel":
            tab1, tab2, tab3= st.tabs(["Cataract", "Entropion", "Epiphora"])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.write("Japanese Chin can develop cataracts as young as 4 years old. Cataracts are usually hereditary, so breeders should not breed Japanese Chin with a family history of cataracts. Surgery to remove the cataracts and restore eyesight is a treatment option.")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Entropion")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Entroption = '''when the eyelids roll inward toward the eye. The fur on the eyelids and the eyelashes then rub against the surface of the eye (the cornea). This is a very painful condition that can lead to corneal ulcers.'''
                    st.markdown(Entroption)
                with st.expander("See More Details"):
                    st.write("Entropion is usually diagnosed in Japanese Chin puppies. Thankfully, this condition can be corrected with surgery. Dogs with a history of entropion should not be bred.")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Epiphora")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Ephiphora =''' an overflow of tears from the eyes. It is a symptom rather than a specific disease and is associated with a variety of conditions. Normally, a thin film of tears is produced to lubricate the eyes and the excess fluid drains into the tear ducts (nasolacrimal ducts) located in the corner of the eye next to the nose. The tear ducts drain tears into the back of the sinuses and down the throat. Epiphora can be caused by either insufficient drainage of tears through the tear ducts, or by an excessive production of tears.'''
                    st.markdown(Ephiphora)
                with st.expander("See More Details"):
                    st.subheader("What are the signs of epiphora?")
                    st.write("The most common clinical signs associated with epiphora are dampness or wetness beneath the eyes, reddish-brown staining of the fur beneath the eyes, odor, skin irritation and skin infection. Many owners report that their dog's face is constantly damp, and they may even see tears rolling off their pet's face. ")
                    st.markdown("---")
                    st.subheader("How is epiphora diagnosed?")
                    st.write("The first step is to determine if there is an underlying cause for the excess tear production. Some of the causes of increased tear production in dogs include conjunctivitis (viral or bacterial), allergies, eye injuries, abnormal eyelashes (distichia or ectopic cilia), corneal ulcers, eye infections, anatomical abnormalities such as rolled in eyelids (entropion) or rolled out eyelids (ectropion), and glaucoma.")
                    st.markdown("---")
                    st.subheader("How is epiphora treated?")
                    st.write("If the nasolacrimal duct is suspected of being blocked, your dog will be anesthetized and a special instrument will be inserted into the duct to flush out the contents. In some cases, the lacrimal puncta or opening may have failed to open during the dog's development, and if this is the case, it can be surgically opened during this procedure. If chronic infections or allergies have caused the ducts to become narrowed, flushing may help widen them. If the cause is related to another eye condition, treatment will be directed at the primary cause which may include surgery.")
        elif breed_label == "Maltese Dog":
            tab1, tab2, tab3= st.tabs(["Glaucoma", "Deafness", "Distichiasis"])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Glaucoma")
                    st.image("https://www.animaleyecare.com.au/images/animal-eye-care/conditions/glaucoma-in-dogs-w.jpg")
                with col2:
                    Glaucoma = '''A disease of the eye in which the pressure within the eye, called intraocular pressure (IOP), is increased. Intraocular pressure is measured using an instrument called a tonometer.'''
                    st.markdown(Glaucoma)
                with st.expander("See More Details"):
                    st.subheader("What is intraocular pressure and how is it maintained?")
                    st.write("The inside of the eyeball is filled with fluid, called aqueous humor. The size and shape of the normal eye is maintained by the amount of fluid contained within the eyeball. The pressure of the fluid inside the front or anterior chamber of the eye is known as intraocular pressure (IOP). Aqueous humor is produced by a structure called the ciliary body. In addition to producing this fluid (aqueous humor), the ciliary body contains the suspensory ligaments that hold the lens in place. Muscles in the ciliary body pull on the suspensory ligaments, controlling the shape and focusing ability of the lens.Aqueous humor contains nutrients and oxygen that are used by the structures within the eye. The ciliary body constantly produces aqueous humor, and the excess fluid is constantly drained from the eye between the cornea and the iris. This area is called the iridocorneal angle, the filtration angle, or the drainage angle.As long as the production and absorption or drainage of aqueous humor is equal, the intraocular pressure remains constant.")
                    st.markdown('---') 
                    st.subheader("What causes glaucoma?")
                    st.write("Glaucoma is caused by inadequate drainage of aqueous fluid; it is not caused by overproduction of fluid. Glaucoma is further classified as primary or secondary glaucoma.")
                    st.write(f"**Primary glaucoma** results in increased intraocular pressure in a healthy eye. Some breeds are more prone than others (see below). It occurs due to inherited anatomical abnormalities in the drainage angle.")
                    st.write(f"**Secondary glaucoma** results in increased intraocular pressure due to disease or injury to the eye. This is the most common cause of glaucoma in dogs. Causes include:")
                    st.write(f"**Uveitis** (inflammation of the interior of the eye) or severe intraocular infections, resulting in debris and scar tissue blocking the drainage angle.")
                    st.write(f"**Anterior dislocation of lens**. The lens falls forward and physically blocks the drainage angle or pupil so that fluid is trapped behind the dislocated lens.")
                    st.write(f"**Tumors** can cause physical blockage of the iridocorneal angle.")
                    st.write(f"**Intraocular bleeding.** If there is bleeding in the eye, a blood clot can prevent drainage of the aqueous humor.")
                    st.write(f"Damage to the lens. Lens proteins leaking into the eye because of a ruptured lens can cause an inflammatory reaction resulting in swelling and blockage of the drainage angle.")
                    st.markdown('---') 
                    st.subheader("What are the signs of glaucoma and how is it diagnosed?")
                    st.write("The most common signs noted by owners are:")
                    st.write(f"**Eye pain**. Your dog may partially close and rub at the eye. He may turn away as you touch him or pet the side of his head.")
                    st.write(f"A **watery discharge** from the eye.")
                    st.write(f"**Lethargy, loss of appetite** or even **unresponsiveness.**")
                    st.write(f"**Obvious physical swelling and bulging of the eyeball** The white of the eye (sclera) looks red and engorged.")
                    st.write(f"The cornea or clear part of the eye may become cloudy or bluish in color.")
                    st.write(f"Blindness can occur very quickly unless the increased IOP is reduced.")
                    st.write(f"**All of these signs can occur very suddenly with acute glaucoma**. In chronic glaucoma they develop more slowly. They may have been present for some time before your pet shows any signs of discomfort or clinical signs.")
                    st.write(f"Diagnosis of glaucoma depends upon accurate IOP measurement and internal eye examination using special instruments. **Acute glaucoma is an emergency**. Sometimes immediate referral to a veterinary ophthalmologist is necessary.")
                    st.markdown('---') 
                    st.subheader("What is the treatment for glaucoma?")
                    st.write("It is important to reduce the IOP as quickly as possible to reduce the risk of irreversible damage and blindness. It is also important to treat any underlying disease that may be responsible for the glaucoma. Analgesics are usually prescribed to control the pain and discomfort associated with the condition. Medications that decrease fluid production and promote drainage are often prescribed to treat the increased pressure. Long-term medical therapy may involve drugs such as carbonic anhydrase inhibitors (e.g., dorzolamide 2%, brand names Trusopt® and Cosopt®) or beta-adrenergic blocking agents (e.g., 0.5% timolol, brand names Timoptic® and Betimol®). Medical treatment often must be combined with surgery in severe or advanced cases. Veterinary ophthalmologists use various surgical techniques to reduce intraocular pressure. In some cases that do not respond to medical treatment or if blindness has developed, removal of the eye may be recommended to relieve the pain and discomfort.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.header("Deafness")
                    st.image("https://www.aaha.org/contentassets/a9895c6d4d55453a8080abd33a77e2e6/blog-2-test---gettyimages-1330594294.png")
                with col2:
                    Deafness = '''An inability to hear, due to many different causes. In Dalmatians, congenital deafness is associated with blue eye color. Deafness may be congenital (present at birth) or acquired as a result of infection, trauma, or degeneration of the cochlea (the organ of hearing).'''
                    st.markdown(Deafness)
                with st.expander("See More Details"):
                    st.write("Deafness present at birth can be inherited or result from toxic or viral damage to the developing unborn puppy. Merle and white coat colors are associated with deafness at birth in dogs and other animals. Dog breeds commonly affected include the Dalmatian, Bull Terrier, Australian Heeler, Catahoula, English Cocker Spaniel, Parson Russell Terrier, and Boston Terrier. The list of affected breeds (now approximately 100) continues to expand and may change due to breed popularity and elimination of the defect through selective breeding.")
                    st.markdown("---")
                    st.write("Acquired deafness may result from blockage of the external ear canal due to longterm inflammation (otitis externa) or excessive ear wax. It may also occur due to a ruptured ear drum or inflammation of the middle or inner ear. Hearing usually returns after these types of conditions are resolved.")
                    st.markdown("---")
                    st.write("The primary sign of deafness is failure to respond to a sound, for example, failure of noise to awaken a sleeping dog, or failure to alert to the source of a sound. Other signs include unusual behavior such as excessive barking, unusual voice, hyperactivity, confusion when given vocal commands, and lack of ear movement. An animal that has gradually become deaf, as in old age, may become unresponsive to the surroundings and refuse to answer the owner’s call.")
                    st.markdown("---")
                    st.write("Deaf dogs do not appear to experience pain or discomfort due to the condition. However, caring for a dog that is deaf in both ears requires more dedication than owning a hearing dog. These dogs are more likely to be startled, which can lead to biting. These dogs are also less protected from certain dangers, such as motor vehicles.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Distichiasis")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/2136//2023_2135i_distichia_eye_6021.jpg")
                with col2:
                    Distichiasis = '''A distichia (plural distichiae) is an extra eyelash that grows from the margin of the eyelid through the duct or opening of the meibomian gland or adjacent to it. Meibomian glands produce lubricants for the eye and their openings are located along the inside edge of the eyelids. The condition in which these abnormal eyelashes are found is called distichiasis.'''
                    st.markdown(Distichiasis)
                with st.expander("See More Details"):
                    st.subheader("What causes distichiasis?")
                    st.write("Sometimes eyelashes arise from the meibomian glands. Why the follicles develop in this abnormal location is not known, but the condition is recognized as a hereditary problem in certain breeds of dogs. Distichiasis is a rare disorder in cats.")
                    st.markdown("---")
                    st.subheader("What breeds are more likely to have distichiasis?")
                    st.write("The more commonly affected breeds include the American Cocker Spaniel, Cavalier King Charles Spaniel, Shih Tzu, Lhasa Apso, Dachshund, Shetland Sheepdog, Golden Retriever, Chesapeake Retriever, Bulldog, Boston Terrier, Pug, Boxer Dog, Maltese, and Pekingese.")
                    st.markdown("---")
                    st.subheader("How is distichiasis diagnosed?")
                    st.write("Distichiasis is usually diagnosed by identifying lashes emerging from the meibomian gland openings or by observing lashes that touch the cornea or the conjunctival lining of the affected eye. A thorough eye examination is usually necessary, including fluorescein staining of the cornea and assessment of tear production in the eyes, to assess the extent of any corneal injury and to rule out other causes of the dog's clinical signs. Some dogs will require topical anesthetics or sedatives to relieve the intense discomfort and allow a thorough examination of the tissues surrounding the eye.")
                    st.markdown("---")
                    st.subheader("How is the condition treated?")
                    st.write("Dogs that are not experiencing clinical signs with short, fine distichia may require no treatment at all. Patients with mild clinical signs may be managed conservatively, through the use of ophthalmic lubricants to protect the cornea and coat the lashes with a lubricant film. Removal of distichiae is no longer recommended, as they often grow back thicker or stiffer, but they may be removed for patients unable to undergo anesthesia or while waiting for a more permanent procedure.")
                    st.markdown("---")
        elif breed_label == "Pekinese":
            tab1, tab2, tab3= st.tabs(["Fold dermatitis", "Inguinal hernia", "Keratitis sicca"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                     st.header("Fold dermatitis")
                     st.image("https://todaysveterinarypractice.com/wp-content/uploads/sites/4/2019/10/2019_1112_SkinFoldDermatitis_Fig2.png")
                with col2:
                     Fold_dermatitis = '''An inflammatory condition of skin folds, induced or aggravated by heat, moisture, maceration, friction, and lack of air circulation.

Intertrigo frequently is worsened by infection, which most commonly is with Candida species. Bacterial, viral, or other fungal infection may also occur.'''
                     st.markdown(Fold_dermatitis)
                with st.expander("See More Details"):
                    st.subheader("Diagnosis")
                    st.write("Basic microbiologic diagnostic studies can be performed to identify a potential causative agent of intertrigo and guide antimicrobial therapy. Potassium hydroxide (KOH) test, Gram stain, or culture is useful to exclude primary or secondary infection and to guide intertrigo therapy. A skin biopsy generally is not required unless the intertrigo is refractory to medical treatment.")
                    st.markdown("---")
                    st.subheader("Treatment of intertrigo")            
                    st.write("Simple intertrigo may be treated with drying agents. Infected intertrigo should be treated with a combination of an appropriate antimicrobial agent (antifungal or antibacterial) and low-potency topical steroid. ")
                    st.markdown("---")
                    st.subheader("Prevention of intertrigo")
                    st.write("During patient instruction, emphasize topics such as weight loss, glucose control (in patients with diabetes), good hygiene, and the need for daily care and monitoring. Additionally, preventative measures to reduce skin-on-skin friction and moisture can help in the management of current intertrigo and prevent future episodes.")
                    st.markdown('---')
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Inguinal hernia")
                    st.image("https://www.wellnessvet.com.hk/wp-content/uploads/2023/05/Service-Inguinal-Hernia-2.jpg")
                with col2:
                    Inguinal_Hernia = '''An inguinal hernia is a condition in which the abdominal contents protrude through the inguinal canal or inguinal ring, an opening which occurs in the muscle wall in the groin area.'''
                    st.markdown(Inguinal_Hernia)
                with st.expander("See More Details"):
                    st.subheader("Causes")
                    st.write("In dogs, inguinal hernias may be acquired (not present at birth but developing later in life) or congenital (present at birth). Factors which predispose a dog to develop an inguinal hernia include trauma, obesity, and pregnancy.")
                    st.write("Most inguinal hernias are uncomplicated and cause no symptoms other than a swelling in the groin area. However, if contents from the abdominal cavity (such as the bladder, a loop of intestines or the uterus) pass through the opening and become entrapped there, the situation can become life-threatening.")
                    st.markdown("---")
                    st.subheader("Diagnosis")
                    st.write("Inguinal hernias can usually be diagnosed by finding the swelling caused by the hernia on a physical examination. However, sometimes contrast radiographs (X-rays) or an abdominal ultrasound are needed to determine which abdominal contents, if any, are entrapped.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("Treatment is surgical correction of the opening and replacement of abdominal contents back into the abdomen if necessary.")
                    st.markdown("---")
                    st.subheader("Prevention")
                    st.write("Because inguinal hernias can be hereditary, dogs with these hernias should not be bred.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Keratitis sicca")
                    st.image("https://images.ctfassets.net/4dmg3l1sxd6g/6ofoq7dPDVcxWBc3WMmwwu/17a5b491521dec9c5558f2b51e88e3b0/dry-eye-in-dogs_figure-4-35281-article.png_-_en", width=200)
                with col2:
                    Keratitis_sicca = '''Dry eye syndrome in dogs, also known as Keratoconjunctivitis Sicca (KCS), involves decreased or inadequate tear production. Tears are important to the lubrication, comfort, and overall health of a dog’s eyes. '''
                    st.markdown(Keratitis_sicca)
                with st.expander("See More Details"):
                    st.subheader("What Is Dry Eye Syndrome in Dogs?")
                    st.write("Tears also contain antibacterial proteins, mucus, white blood cells to fight infection, and other enzymes to help keep the eyes clear and free of debris, infection, and irritations. ")
                    st.markdown("---")
                    st.subheader("Symptoms of Dry Eye Syndrome in Dogs")
                    st.write("Dogs with dry eye syndrome can exhibit one or many of the following symptoms: ")
                    st.write("Red, inflamed, irritated, and painful eyes ")
                    st.write("Redness and swelling of the conjunctiva, or the tissues around the eye ")
                    st.write("Frequent squinting and blinking ")
                    st.write("Dryness on the surface of the cornea—the outer clear dome-shaped part of the eye ")
                    st.write("Mucous-like discharge on the cornea (may be yellow or green if a secondary bacterial infection is present) ")
                    st.write("Obvious defects and irregularities of the cornea, including increased vascularization (abnormal growth of blood vessels to the injured area) and pigmentation as the eye attempts to heal and protect itself ")
                    st.write("Possible vision impairment and blindness ")
                    st.markdown("---")
                    st.subheader("Causes of Dry Eye Syndrome in Dogs")
                    st.write("The cause of dry eye syndrome in a dog can be due to one or a few underlying conditions. Your veterinarian will be able to determine what may have caused your pet’s diagnosis based on the dog’s medical history and an exam. Some of the underlying causes may be due to: ")
                    st.write("Immune system dysfunction: Most cases of dry eye syndrome in dogs are caused by the immune system attacking and destroying the lacrimal and third eyelid gland. Unfortunately, veterinarians do not know why this happens.  ")
                    st.write("Medications: Certain drugs can cause dry eye syndrome as a side effect, usually very shortly after a dog starts taking these medications. This type of dry eye syndrome can be temporary and may go away once the medication is discontinued. However, permanent damage can be done, and there is no way to predict which animals will have dry eye syndrome or how long it will last. Be sure to talk with your vet about possible side effects of all medications.  ")
                    st.write("Genes: Congenital alacrimia is a genetic form of dry eye syndrome and occurs in some breeds, most notably Yorkshire terriers. This is typically noticed in only one eye.  ")
                    st.write("Endocrine conditions: Some systemic disease (such as hypothyroidism, diabetes, and Cushing’s disease) frequently decrease tear production. ")
                    st.write("Infectious diseases: Canine distemper virus, leishmaniasis, and chronic blepharoconjunctivitis can all lead to dry eye syndrome. ")
                    st.write("Medical procedures: A common abnormality of dogs is a prolapsed third eyelid gland (more commonly known as cherry eye). While it is not recommended, some surgeons remove the gland entirely, leading to permanent decreased tear production. Local radiation for tumors can also cause permanent damage to the lacrimal and third eyelid glands. ")
                    st.write("Neurological problems: Loss of nerve function to the glands (commonly secondary to an inner ear infection) can decrease or stop production of tears. ")
                    st.write("Traumatic injury: Dry eye syndrome can occur with damage to the glands after severe inflammation or injury (such as from wounds or car accidents). ")
                    st.write("Transient causes: Anesthesia causes a temporary loss of tear production, as does the medication atropine. Once these are removed, tear production normally returns.  ")
                    st.markdown("---")
                    st.subheader("How Veterinarians Diagnose Dry Eye Syndrome in Dogs")
                    st.write("Vets use the Schirmer Tear Test (STT) to diagnose dry eye syndrome and measure aqueous tear production in dogs. This is a simple, painless test involving a strip of special paper placed in the lower eyelid. The moisture and tears from the eye wick onto the paper for 60 seconds. At the end of that time, the vet measures the tear production on the paper. For test results, more than 15 millimeters of tear production per minute is normal, while less than 10 millimeters indicates dry eye syndrome. Your vet may repeat the test to confirm the diagnosis. After the STT, your vet may also perform a fluorescein stain test to look for corneal ulcers. The stain makes an ulcer glow bright green under a black light. The vet may also use a test of intraocular pressure to look for inflammation or glaucoma. These conditions are common with dry eye and important to diagnose and treat at the same time.")
                    st.markdown("---")
                    st.subheader("Treatment of Dry Eye Syndrome in Dogs")
                    st.write("Lacrimostimulants: Most commonly, vets prescribe ophthalmic cyclosporine (a class of medications) or tacrolimus to stimulate tear production. Cyclosporine, when applied in the eye, keeps the immune system from harming the lacrimal and third eyelid glands, thus allowing tear restoration. Tacrolimus is typically used only if cyclosporine fails.  ")
                    st.write("Lacrimomimetics: Artificial tears moisten the surface of the eye, improve comfort, and help flush debris and allergens. These eye lubricants are essential to use with primary medications for dry eye syndrome, like cyclosporine, especially early in the treatment process when tear production hasn’t fully recovered. Only use artificial tears if your vet directs you. ")
                    st.write("Antibiotics: Bacterial infections and corneal ulcerations may also require broad-spectrum topical antibiotics. Dogs whose dry eye syndrome is related to the nervous system are treated with pilocarpine, which stimulates glandular secretion.  ")
                    st.write("Surgery: Dogs who don’t respond to treatment may require a surgery called parotid duct transposition, which carefully redirects saliva glands in the dog’s mouth to the eye, so that saliva can be used as tears. ")
                    st.markdown("---")
        elif breed_label == "Shih-Tzu":
            tab1, tab2, tab3= st.tabs(["Epiphora", "Dwarfism", "Cataract"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Epiphora")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Ephiphora =''' an overflow of tears from the eyes. It is a symptom rather than a specific disease and is associated with a variety of conditions. Normally, a thin film of tears is produced to lubricate the eyes and the excess fluid drains into the tear ducts (nasolacrimal ducts) located in the corner of the eye next to the nose. The tear ducts drain tears into the back of the sinuses and down the throat. Epiphora can be caused by either insufficient drainage of tears through the tear ducts, or by an excessive production of tears.'''
                    st.markdown(Ephiphora)
                with st.expander("See More Details"):
                    st.subheader("What are the signs of epiphora?")
                    st.write("The most common clinical signs associated with epiphora are dampness or wetness beneath the eyes, reddish-brown staining of the fur beneath the eyes, odor, skin irritation and skin infection. Many owners report that their dog's face is constantly damp, and they may even see tears rolling off their pet's face. ")
                    st.markdown("---")
                    st.subheader("How is epiphora diagnosed?")
                    st.write("The first step is to determine if there is an underlying cause for the excess tear production. Some of the causes of increased tear production in dogs include conjunctivitis (viral or bacterial), allergies, eye injuries, abnormal eyelashes (distichia or ectopic cilia), corneal ulcers, eye infections, anatomical abnormalities such as rolled in eyelids (entropion) or rolled out eyelids (ectropion), and glaucoma.")
                    st.markdown("---")
                    st.subheader("How is epiphora treated?")
                    st.write("If the nasolacrimal duct is suspected of being blocked, your dog will be anesthetized and a special instrument will be inserted into the duct to flush out the contents. In some cases, the lacrimal puncta or opening may have failed to open during the dog's development, and if this is the case, it can be surgically opened during this procedure. If chronic infections or allergies have caused the ducts to become narrowed, flushing may help widen them. If the cause is related to another eye condition, treatment will be directed at the primary cause which may include surgery.")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Dwarfism")
                    st.image("https://sacriverkennel.com/wp-content/uploads/2014/09/Doll_001.jpg")
                with col2:
                    Dwarfism = '''Dwarfism in dogs encompasses several growth and developmental conditions, usually inherited, which result in a dog being smaller than it should be. Dwarfism often results in health issues and a reduction in life span. '''
                    st.markdown(Dwarfism)
                with st.expander("See More Details"):
                    st.subheader("What are the causes of dwarfism in dogs?")
                    st.write("There are several causes of dwarfism in dogs, with pituitary dwarfism and achondroplasia the commonest. Dogs with pituitary dwarfism have an underdeveloped pituitary gland and a deficiency of growth hormone. It is a genetic condition and is seen most often in German Shepherds. Affected dogs are small but their body remains in the correct proportions.")
                    st.write("Achondroplasia is a developmental abnormality affecting bone and cartilage. Again it is a genetic condition but affected dogs tend to have short limbs that are out of proportion to the rest of their body. They may also have other abnormalities and because it is a hereditary condition, it is recommended that affected dogs should not be used for breeding.")
                    st.markdown("---")
                    st.subheader("How can you tell if a dog has dwarfism?")
                    st.write("The signs of dwarfism will vary depending on the underlying cause, but here are some signs to look out for:")
                    st.write("Symptoms of dwarfism in dogs")
                    st.write("Bones look shorter than normal, Enlarged joints, Abnormal bone shape, Larger head than normal, Shorter nose with an undershot jaw, Crooked teeth, Lack of growth, Spinal deviation (either side), Bowing of forelimbs (leaning out sideways), Heart issues, Fear and dog aggression")
                    st.markdown('---') 
                    st.subheader("Diagnosing dwarfism in dogs")
                    st.write("For a diagnosis of dwarfism, the vet will usually examine your dog and may carry out X-rays to look at bone development. Make note of any more abnormalities you’ve spotted. To rule out other conditions and further confirm if a dog has dwarfism, the vet may want to take blood and send it to the lab for further testing.")
                    st.markdown("---")
                    st.subheader("Can dwarfism in dogs be treated?")
                    st.write("Though there is no cure for dwarfism, whatever the cause, your vet may suggest medication to help manage the condition together with pain relief if required. For example, it is possible to treat and manage pituitary dwarfism in dogs with prescribed dosages of growth hormone. In mature dogs, spaying or neutering could also help. It’s important to talk to a vet first if you suspect that your dog has dwarfism.")
                    st.markdown("---")
                    st.subheader("How long do dogs live with dwarfism? ")
                    st.write("Depending on the severity of the case, some dogs with dwarfism can live relatively normal lives. Other dogs, however, unfortunately, won’t live past 5 years of age. This does depend on the breed and type of dwarfism – your vet will be able to provide more specifics for your dog.")
                    st.markdown("---")
                    st.subheader("Caring for dogs with dwarfism")
                    st.write("Giving your dog a good quality of life is highly important. Dwarfism in dogs can cause certain health issues that will need to be managed to ensure your pooch is as happy as possible. One thing to bear in mind with achondroplasia is that affected dogs tend to be prone to arthritis. Another thing to note is obesityas carrying extra weight puts additional strain on arthritic joints– making sure your dog has a healthy diet and a good amount of exercise is important here. Some pain medication can help too but make sure a vet is consulted for the best course of action.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.write("A common cause of blindness in older Shih Tzus. We'll watch for the lenses of his eyes to become more opaque—meaning they look cloudy instead of clear—when examined. Many dogs adjust well to losing their vision and get along just fine.")
                    st.write("Shih Tzu eye problems are not uncommon. Unfortunately, these problems cause discomfort, irritations and in some extreme circumstances, vision loss. However, with attention and care on a daily basis, you can recognize many of the symptoms before it develops into a serious issue.")
                    st.write("Cataract growth is genetically inherited in pure bred Shih Tzu and cause a white film to cover the eye, eventually causing blindness when left untreated.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("To remove these cataracts, surgery is implemented usually with successful results. Although this operation will remove the chance of blindness, he may subsequently experience a slight reduction in vision.")
                    st.markdown("---")
                    st.subheader("Prevension")
                    st.write("Unfortunately you can't really prevent cataracts but you may be able to slow the progression with a good healthy diet and avoidance of toxins.")
                    st.markdown("---")
        elif breed_label == "Blenheim Spaniel":
            tab1, tab2, tab3= st.tabs(["Cataract", "Cleft palate", "Retinal dysplasia"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.write("The cavalier King Charles spaniel has a strong breed predisposition to develop congenital, early-onset juvenile cataracts, which appear by 6 months of age in both eyes and progress to complete cataracts and total blindness by between ages 2 and 4 years. This form of cataract usually is combined with other ocular disorders. Older cavaliers also are prone to develop a non-congenital form of cataract, usually in both eyes, which also are progressive, and may be expected to form in cavaliers as old as 7 years of age.")
                    st.markdown("---")
                    st.subheader("Symptoms")
                    st.write("Cataracts usually are discovered first by noticing discoloration in the cavalier's eyes. The center of the eye will appear light gray or yellowish, or white. Also, the owner likely will observe the dog having visual difficulties. The cavalier may bump into things, including familiar objects, or appear tentative about moving up or down on stair steps.")
                    st.markdown("---")
                    st.subheader("Diagnosis")
                    st.write("Cataracts are visible using an ophthalmoscope and may be discovered during a routine eye examination.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("If the cataracts seriously affect the dog's vision, they may be removed by surgery. Phacoemulsification is a cataract surgery method in which the internal lens is emulsified with an ultrasonic probe and the sucked from the eye. Some ophthalmologist surgeons also will insert an artificial replacement lens which reportedly will restore near-normal vision. Implant replacement surgeries usually take about an hour per eye and reportedly are about 90% of them have been successful for dogs deemed to be good candidates for the surgery")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cleft Palate") 
                    st.image("https://www.bpmcdn.com/f/files/victoria/import/2021-06/25618240_web1_210605-CPL-SPCA-Lock-In-Fore-Love-Baby-Snoot-Chilliwack_4.jpg", width=200)
                with col2:
                    Cleft_Palate = '''A condition where the roof of the mouth is not closed 
                                and the inside of the nose opens into the mouth. It occurs due to a failure of the roof of the mouth to close during 
                                development in the womb. This results in a hole between the mouth and the nasal cavity. 
                                The defect can occur in the lip (primary cleft palate) or along the roof of the mouth (secondary cleft palate).
                            '''
                    st.markdown(Cleft_Palate)
                with st.expander("See More details"):
                    st.subheader("Cleft palate in puppies Prognosis")
                    st.write("A cleft palate is generally detected by visual examination of newborn puppies by the veterinary surgeon or breeder. Cleft palate of the lip or hard palate are easy to see, but soft palate defects can sometimes require sedation or general anaesthesia to visualise. Affected puppies will often have difficulty suckling and swallowing. This is often seen as coughing, gagging, and milk bubbling from the pup’s nose. In less severe defects, more subtle signs such as sneezing, snorting, failure to grow, or sudden onset of breathing difficulty (due to aspiration of milk or food) can occur.")
                    st.markdown("---")
                    st.subheader("Treatment for cleft palate in puppies")
                    st.write("Treatment depends on the severity of the condition, the age at which the diagnosis is made, and whether there are complicating factors, such as aspiration pneumonia.")
                    st.write("Small primary clefts of the lip and nostril of the dog are unlikely to cause clinical problems.")
                    st.write("Secondary cleft palates in dogs require surgical treatment to prevent long-term nasal and lung infections and to help the puppy to feed effectively. The surgery involves either creating a single flap of healthy tissue and overlapping it over the defect or creating a ‘double flap’, releasing the palate from the inside of the upper teeth, and sliding it to meet in the middle over the defect.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Retinal Dysplasia")
                    st.image("https://cavalierhealth.org/images/eye%20exam.png")
                with col2:
                    Retinal_Dysplasia = '''The most serious eye defects that afflict high percentages of cavaliers are forms of retinal dysplasia (RD), according to the American College of Veterinary Ophthalmologists (ACVO).* Retinal dysplasia is a congenital malformation of the retina. It occurs when the two layers of the retina do not form together properly.'''
                    st.markdown(Retinal_Dysplasia)
                with st.expander("See More Details"):
                    st.subheader("Diagnosis")
                    st.write("Upon examination, the ophthalmologist can tell the degree of severity of the dysplasia. Most cases of retinal dysplasia do not progress after puppyhood, and the ophthalmologist may be able to predict the extent to which the dysplasia will interfere with the dog's field of vision. The cause of most cases of retinal dysplasia in cavaliers is genetic. Of all purebred dogs, multifocal retinal dysplasia (MRD) and geographic retinal dysplasia are most commonly found in the cavalier King Charles spaniel")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("There is no treatment for this condition. While there is ongoing research, there is nothing yet to prevent this condition from occurring. Prevention comes in the form of not breeding your dog if he has this condition as it will pass the genes on to the next generation.")
                    st.markdown("---")
        elif breed_label == "Papillon":
            tab1, tab2, tab3= st.tabs(["Patella luxation", "Follicular dysplasia", "Deafness"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Patella luxation")
                    st.image("https://www.lifelearn-cliented.com//cms/resources/body/836/patellar_luxation_2018_ek_db-01.scaler.jpg")
                with col2:
                    Patella_luxation = '''The knee joint connects the femur, (thighbone) and the tibia (shinbone). The patella (kneecap) is normally located in a groove called the trochlear groove, found at the end of the femur. The term luxating means out of place or dislocated. Therefore, a luxating patella is a kneecap that 'pops out' or moves out of its normal location.'''
                    st.markdown(Patella_luxation)
                with st.expander("See More Details"):
                    st.subheader("What causes a patellar luxation?")
                    st.write("The kneecap sits underneath a ligament called the patellar ligament. This ligament attaches the large thigh muscles to a point on the center front of the shin bone (tibia). When the thigh muscles contract, the force is transmitted through the patellar ligament, pulling on the shin bone. This results in extension or straightening of the knee. The patella slides up and down in its groove (trochlear groove) and helps keep the patellar ligament in place during this movement. In some dogs, especially ones that are bowlegged, the patella may luxate because the point of attachment of the patellar ligament is not in the center of the shinbone. In these cases, it is almost always located too far toward the middle of the body or the inside of the leg. As the thigh muscles contract, the force applied to the patella pulls it to the inside of the knee. After several months or years of this abnormal movement, the inner side of the groove in the femur may wear down. Once this happens, the patella is then free to dislocate or slide toward the inside of the knee.")
                    st.markdown("---")
                    st.subheader("Can a luxating patella cause long-term problems?")
                    st.write("This depends upon the grade of the luxation and whether both legs are affected to the same degree. The higher the grade, the more likely your dog will develop long-term problems. Some dogs, especially with Grade I patellar luxation, can tolerate this condition for many years, even for their entire lives; however, as the dog ages, arthritis develops and results not only in decreased mobility but joint pain as well. Once arthritis develops, it cannot be reversed. In addition, patellar luxation predisposes the knee to other injuries, especially torn cruciate ligaments.")
                    st.markdown("---")
                    st.subheader("Can a luxating patella be corrected?")
                    st.write("Surgery should be performed if your dog has recurrent or persistent lameness or if other knee injuries occur secondary to the luxating patella. ")
                    st.markdown("---")
                    st.subheader("What is the prognosis?")
                    st.write("If your veterinarian performs surgery before arthritis or another knee injury occurs, the prognosis is excellent. Your dog should regain full use of her leg. However, if arthritis has already developed in the knee joint, your dog may experience intermittent pain in the leg and it may progress. The higher the grade of patellar luxation, the higher the likelihood of reoccurrence postoperatively. Prescription anti-inflammatories, joint supplements, and/or therapeutic mobility diets may slow the progression of arthritis and help control any discomfort. Weight reduction is also recommended for overweight dogs. Post-operative physiotherapy may be recommended. Your veterinarian can help you determine the best post-operative plan for your dog.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Follicular dysplasia")
                    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Color_dilution_alopecia_2.jpg/220px-Color_dilution_alopecia_2.jpg")
                with col2:
                    Follicular_dysplasia = '''Follicular dysplasia is a condition caused by an abnormality in the canine hair follicle. It typically manifests as hair loss or abnormal hair growth that progresses over an animal's lifetime.'''
                    st.markdown(Follicular_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("Symptoms and Identification")
                    st.write("For black hair follicular dysplasia, symptoms include a progressively worsening, permanent hair loss over black areas of the skin that begins at about four weeks of age. Scaling and flaking of the skin is common, as are secondary infections that can impart an odor to the skin and elicit itching. For other types of follicular dysplasia, the symptoms will differ by breed")
                    st.write("Though the condition can often be diagnosed by breed and symptoms alone, skin biopsy is strongly recommended for definitive diagnosis.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("The disease is not treatable. Management of the scaling and secondary infections is usually undertaken via supplements, shampoos, topical applications and topical antimicrobials when necessary")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Deafness")
                    st.image("https://www.aaha.org/contentassets/a9895c6d4d55453a8080abd33a77e2e6/blog-2-test---gettyimages-1330594294.png")
                with col2:
                    Deafness = '''An inability to hear, due to many different causes. In Dalmatians, congenital deafness is associated with blue eye color. Deafness may be congenital (present at birth) or acquired as a result of infection, trauma, or degeneration of the cochlea (the organ of hearing).'''
                    st.markdown(Deafness)
                with st.expander("See More Details"):
                    st.write("Deafness present at birth can be inherited or result from toxic or viral damage to the developing unborn puppy. Merle and white coat colors are associated with deafness at birth in dogs and other animals. Dog breeds commonly affected include the Dalmatian, Bull Terrier, Australian Heeler, Catahoula, English Cocker Spaniel, Parson Russell Terrier, and Boston Terrier. The list of affected breeds (now approximately 100) continues to expand and may change due to breed popularity and elimination of the defect through selective breeding.")
                    st.markdown("---")
                    st.write("Acquired deafness may result from blockage of the external ear canal due to longterm inflammation (otitis externa) or excessive ear wax. It may also occur due to a ruptured ear drum or inflammation of the middle or inner ear. Hearing usually returns after these types of conditions are resolved.")
                    st.markdown("---")
                    st.write("The primary sign of deafness is failure to respond to a sound, for example, failure of noise to awaken a sleeping dog, or failure to alert to the source of a sound. Other signs include unusual behavior such as excessive barking, unusual voice, hyperactivity, confusion when given vocal commands, and lack of ear movement. An animal that has gradually become deaf, as in old age, may become unresponsive to the surroundings and refuse to answer the owner’s call.")
                    st.markdown("---")
                    st.write("Deaf dogs do not appear to experience pain or discomfort due to the condition. However, caring for a dog that is deaf in both ears requires more dedication than owning a hearing dog. These dogs are more likely to be startled, which can lead to biting. These dogs are also less protected from certain dangers, such as motor vehicles.")
                    st.markdown("---")
        elif breed_label == "Toy Terrier":
            tab1, tab2, tab3= st.tabs(["Demodicosis", "Atopic dermatitis", "Cataract"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Demodicosis")
                    st.image("https://images.wagwalkingweb.com/media/articles/dog/demodex-/demodex-.jpg?auto=compress&fit=max&width=640")
                with col2:
                    Demodicosis = '''A kind of skin disease (mange) caused by microscopic Demodex canis mites 
                        living within the skin layers and producing an immunodeficiency syndrome'''
                    st.markdown(Demodicosis)
                with st.expander("See More Details"):
                    st.subheader("What is Demodex?")
                    st.write("Demodex, also known as demodectic mange, in dogs is a mite infestation on your dog’s skin. The mites are tiny, eight legged, cigar shaped, and feed in the hair follicles and oil glands of the skin. Most cases of demodex are self-limiting, meaning your dog is able to stop the growth and reproduction of the demodex mites and will also repair the damage that was done by the mites.")
                    st.markdown("---")
                    st.subheader("Symptoms of Demodex in Dogs?")
                    st.write("When Demodex first appears, it may just look like a small spot of hair loss, possibly from rubbing the area. However, if you notice any crusting on the skin or the hair loss spreads contact your veterinarian for an appointment.")
                    st.markdown("---")
                    st.subheader("Causes of Demodex in Dogs")
                    st.write("Demodex is most common in puppies and dogs that have immature immune systems. The mites will multiply uncontrollably when your dog’s immune system is immature or weak and unable to properly dispose of the excessive mites. Most adult dogs will be able to fight off the excessive mites without needing medical intervention. Older dogs may also show symptoms of demodex as their immune systems begin to decline and with age.")
                    st.markdown("---")
                    st.subheader("Diagnosis of Demodex in Dogs")
                    st.write("Your veterinarian will begin by taking a complete medical history on your dog. They will also ask you about any changes in diet or environment. Then, your veterinarian will complete a full physical examination on your dog, paying close attention to any bald spots or noticeable lesions. Your veterinarian will do a complete blood count and will also do a skin scraping of an affected area. The skin scraping will be placed under a microscope and your veterinarian will look for mites. Demodex canis mites are fairly easy to spot under the microscope. If your dog is a mature dog, your veterinarian may also search for the reason the Demodex canis mites were able to multiply uncontrollably. There is usually an underlying cause that is suppressing the immune system and sometimes it is extremely difficult to find what that cause is.")
                    st.markdown("---")
                    st.subheader("Treatment of Demodex in Dogs")
                    st.write("Once your veterinarian has diagnosed demodex they will begin treatments to get rid of the overgrowth of mites. Anti-mite creams can be used as well as anti-inflammatory creams and corticosteroid creams. Your veterinarian may also recommend using benzoyl peroxide on larger areas. Your veterinarian will probably trim the hair around the affected areas. This will allow the prescribed creams to work more effectively on the affected areas")
                    st.write("Some cases of demodex may require the use of anti-parasitic medications. Your veterinarian will prescribe the medications they feel will work best on your dog. Antibiotics may also be used in cases where bacterial infections from the demodex have occurred.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Atopic dermatitis")
                    st.image("https://images.ctfassets.net/82d3r48zq721/6jiyt1463nRXCZa451La8T/5114391b07c7760c8b9068224869ae42/Dog-with-skin-allergy-atopic-dematitis-biting-and-scratching-himself_resized.jpg?w=800&h=531&q=50&fm=webp")
                with col2:
                    Atopic_dermatitis = '''Atopic dermatitis (or atopy) is an inflammatory, chronic skin condition associated with environmental allergies and is the second most common allergic skin condition diagnosed in dogs.'''
                    st.markdown(Atopic_dermatitis)
                with st.expander("See More Details"):
                    st.subheader("Signs & symptoms of atopic dermatitis in dogs")
                    st.write("A dog with atopic dermatitis will usually show signs and symptoms between 3 months to 6 years of age. It’s not as common for dogs over the age of 7 to develop atopic dermatitis, although a new environment can trigger new allergens. Atopic dermatitis often begins as a mild condition with symptoms not becoming clinically visible before the third year.")
                    st.markdown("---")
                    st.subheader("What causes atopic dermatitis in dogs?")
                    st.write("Atopic dermatitis is a genetic disease that is predisposed in some breeds more than others. For that reason, dogs diagnosed with the condition should not be bred. The cause is unknown, but a general understanding of the anatomy of the skin is vital in understanding what happens to a dog when the skin becomes irritated and inflamed as a result of allergens in the environment. A case of atopic dermatitis can be painful and uncomfortable for a dog.")
                    st.markdown("---")
                    st.subheader("Diagnosing canine atopic dermatitis")
                    st.write("Symptoms of atopic dermatitis are similar to other skin conditions, which can make it difficult to diagnose. Uncovering the cause may take time and is often a process of elimination. Along with a full medical examination, which includes a look at the dog’s complete medical history, additional allergy testing may be done. In some cases, your veterinarian may perform a blood test (serological allergy test) to determine the presence of an antibody called IgE to specific allergens. An increase in an allergen-specific IgE usually means there is an overreaction to that allergen in the body.")
                    st.markdown("---")
                    st.subheader("Treatment for atopic dermatitis in dogs")
                    st.write("One of the first steps is eliminating or reducing exposure to the allergens causing dermatitis. If you are unable to identify the irritants, use a process of elimination by removing the environmental factors that have the potential to trigger an outbreak. Diet, bedding, even the general environment in which the dog is exposed to may need to be changed.")
                    st.write("For dogs with a severe case of atopic dermatitis, removing and changing specific factors might not be enough. Oral corticosteroids can be given to control or reduce the itching and swelling, but there are side effects associated with steroids, so it’s important to administer as directed by your veterinarian. There are also other non-steroidal drugs that your veterinarian might prescribe to alleviate the discomfort.")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    
        elif breed_label == "Rhodesian Ridgeback":
            tab1, tab2, tab3= st.tabs(["Atopic dermatitis", "Cataract", "Hip dysplasia"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Atopic dermatitis")
                    st.image("https://images.ctfassets.net/82d3r48zq721/6jiyt1463nRXCZa451La8T/5114391b07c7760c8b9068224869ae42/Dog-with-skin-allergy-atopic-dematitis-biting-and-scratching-himself_resized.jpg?w=800&h=531&q=50&fm=webp")
                with col2:
                    Atopic_dermatitis = '''Atopic dermatitis (or atopy) is an inflammatory, chronic skin condition associated with environmental allergies and is the second most common allergic skin condition diagnosed in dogs.'''
                    st.markdown(Atopic_dermatitis)
                with st.expander("See More Details"):
                    st.subheader("Signs & symptoms of atopic dermatitis in dogs")
                    st.write("A dog with atopic dermatitis will usually show signs and symptoms between 3 months to 6 years of age. It’s not as common for dogs over the age of 7 to develop atopic dermatitis, although a new environment can trigger new allergens. Atopic dermatitis often begins as a mild condition with symptoms not becoming clinically visible before the third year.")
                    st.markdown("---")
                    st.subheader("What causes atopic dermatitis in dogs?")
                    st.write("Atopic dermatitis is a genetic disease that is predisposed in some breeds more than others. For that reason, dogs diagnosed with the condition should not be bred. The cause is unknown, but a general understanding of the anatomy of the skin is vital in understanding what happens to a dog when the skin becomes irritated and inflamed as a result of allergens in the environment. A case of atopic dermatitis can be painful and uncomfortable for a dog.")
                    st.markdown("---")
                    st.subheader("Diagnosing canine atopic dermatitis")
                    st.write("Symptoms of atopic dermatitis are similar to other skin conditions, which can make it difficult to diagnose. Uncovering the cause may take time and is often a process of elimination. Along with a full medical examination, which includes a look at the dog’s complete medical history, additional allergy testing may be done. In some cases, your veterinarian may perform a blood test (serological allergy test) to determine the presence of an antibody called IgE to specific allergens. An increase in an allergen-specific IgE usually means there is an overreaction to that allergen in the body.")
                    st.markdown("---")
                    st.subheader("Treatment for atopic dermatitis in dogs")
                    st.write("One of the first steps is eliminating or reducing exposure to the allergens causing dermatitis. If you are unable to identify the irritants, use a process of elimination by removing the environmental factors that have the potential to trigger an outbreak. Diet, bedding, even the general environment in which the dog is exposed to may need to be changed.")
                    st.write("For dogs with a severe case of atopic dermatitis, removing and changing specific factors might not be enough. Oral corticosteroids can be given to control or reduce the itching and swelling, but there are side effects associated with steroids, so it’s important to administer as directed by your veterinarian. There are also other non-steroidal drugs that your veterinarian might prescribe to alleviate the discomfort.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")

        elif breed_label == "Afghan Hound":
            tab1, tab2, tab3= st.tabs(["Anesthetic idiosyncracy", "Cataract", "Glaucoma"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Anesthetic idiosyncracy")
                    st.image("https://www.mdpi.com/animals/animals-14-00822/article_deploy/html/images/animals-14-00822-g001-550.jpg")
                with col2:
                    Anesthetic_idiosyncracy = '''A condition where an individual has an abnormal response to commonly used anesthetics sometimes leading to death. Idiosyncratic means there is no good explanation or way to predict this.'''
                    st.markdown(Anesthetic_idiosyncracy)
                with st.expander("See More Details"):
                    st.subheader("Symptoms")
                    st.write(" An abnormal, unreliable response to commonly used anaesthetics. In severe cases it can lead to cardiac and/or respiratory arrest during the surgical procedure with the danger of a fatal outcome. Unfortunately, this reaction is completely unpredictable and there is no certain way to predict or determine this kind of response.")
                    st.markdown("---")
                    st.subheader("Disease Cause")
                    st.write("It is believed to be caused by the incapability of the liver to properly metabolise anaesthetic agents.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Glaucoma")
                    st.image("https://www.animaleyecare.com.au/images/animal-eye-care/conditions/glaucoma-in-dogs-w.jpg")
                with col2:
                    Glaucoma = '''A disease of the eye in which the pressure within the eye, called intraocular pressure (IOP), is increased. Intraocular pressure is measured using an instrument called a tonometer.'''
                    st.markdown(Glaucoma)
                with st.expander("See More Details"):
                    st.subheader("What is intraocular pressure and how is it maintained?")
                    st.write("The inside of the eyeball is filled with fluid, called aqueous humor. The size and shape of the normal eye is maintained by the amount of fluid contained within the eyeball. The pressure of the fluid inside the front or anterior chamber of the eye is known as intraocular pressure (IOP). Aqueous humor is produced by a structure called the ciliary body. In addition to producing this fluid (aqueous humor), the ciliary body contains the suspensory ligaments that hold the lens in place. Muscles in the ciliary body pull on the suspensory ligaments, controlling the shape and focusing ability of the lens.Aqueous humor contains nutrients and oxygen that are used by the structures within the eye. The ciliary body constantly produces aqueous humor, and the excess fluid is constantly drained from the eye between the cornea and the iris. This area is called the iridocorneal angle, the filtration angle, or the drainage angle.As long as the production and absorption or drainage of aqueous humor is equal, the intraocular pressure remains constant.")
                    st.markdown('---') 
                    st.subheader("What causes glaucoma?")
                    st.write("Glaucoma is caused by inadequate drainage of aqueous fluid; it is not caused by overproduction of fluid. Glaucoma is further classified as primary or secondary glaucoma.")
                    st.write(f"**Primary glaucoma** results in increased intraocular pressure in a healthy eye. Some breeds are more prone than others (see below). It occurs due to inherited anatomical abnormalities in the drainage angle.")
                    st.write(f"**Secondary glaucoma** results in increased intraocular pressure due to disease or injury to the eye. This is the most common cause of glaucoma in dogs. Causes include:")
                    st.write(f"**Uveitis** (inflammation of the interior of the eye) or severe intraocular infections, resulting in debris and scar tissue blocking the drainage angle.")
                    st.write(f"**Anterior dislocation of lens**. The lens falls forward and physically blocks the drainage angle or pupil so that fluid is trapped behind the dislocated lens.")
                    st.write(f"**Tumors** can cause physical blockage of the iridocorneal angle.")
                    st.write(f"**Intraocular bleeding.** If there is bleeding in the eye, a blood clot can prevent drainage of the aqueous humor.")
                    st.write(f"Damage to the lens. Lens proteins leaking into the eye because of a ruptured lens can cause an inflammatory reaction resulting in swelling and blockage of the drainage angle.")
                    st.markdown('---') 
                    st.subheader("What are the signs of glaucoma and how is it diagnosed?")
                    st.write("The most common signs noted by owners are:")
                    st.write(f"**Eye pain**. Your dog may partially close and rub at the eye. He may turn away as you touch him or pet the side of his head.")
                    st.write(f"A **watery discharge** from the eye.")
                    st.write(f"**Lethargy, loss of appetite** or even **unresponsiveness.**")
                    st.write(f"**Obvious physical swelling and bulging of the eyeball** The white of the eye (sclera) looks red and engorged.")
                    st.write(f"The cornea or clear part of the eye may become cloudy or bluish in color.")
                    st.write(f"Blindness can occur very quickly unless the increased IOP is reduced.")
                    st.write(f"**All of these signs can occur very suddenly with acute glaucoma**. In chronic glaucoma they develop more slowly. They may have been present for some time before your pet shows any signs of discomfort or clinical signs.")
                    st.write(f"Diagnosis of glaucoma depends upon accurate IOP measurement and internal eye examination using special instruments. **Acute glaucoma is an emergency**. Sometimes immediate referral to a veterinary ophthalmologist is necessary.")
                    st.markdown('---') 
                    st.subheader("What is the treatment for glaucoma?")
                    st.write("It is important to reduce the IOP as quickly as possible to reduce the risk of irreversible damage and blindness. It is also important to treat any underlying disease that may be responsible for the glaucoma. Analgesics are usually prescribed to control the pain and discomfort associated with the condition. Medications that decrease fluid production and promote drainage are often prescribed to treat the increased pressure. Long-term medical therapy may involve drugs such as carbonic anhydrase inhibitors (e.g., dorzolamide 2%, brand names Trusopt® and Cosopt®) or beta-adrenergic blocking agents (e.g., 0.5% timolol, brand names Timoptic® and Betimol®). Medical treatment often must be combined with surgery in severe or advanced cases. Veterinary ophthalmologists use various surgical techniques to reduce intraocular pressure. In some cases that do not respond to medical treatment or if blindness has developed, removal of the eye may be recommended to relieve the pain and discomfort.")
                    st.markdown("---")

        elif breed_label == "Basset":
            tab1, tab2, tab3= st.tabs(["Achondroplasia/Dwarfism", "Acute moist dermatitis", "Bloat"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Achondroplasia")
                    st.image("https://cdn.shopify.com/s/files/1/0535/2738/0144/files/Dutch_Blog_Post_1750_1_graphic_1_Copy_45_1024x1024.jpg?v=1668638886")
                with col2:
                    Achondroplasia = '''A developmental abnormality affecting bone and cartilage. Again it is a genetic condition but affected dogs tend to have short limbs that are out of proportion to the rest of their body.'''
                    st.markdown(Achondroplasia)
                with st.expander("See More Details"):
                    st.subheader("How can you tell if a dog has dwarfism?")
                    st.write("The signs of dwarfism will vary depending on the underlying cause, but here are some signs to look out for:")
                    st.write("Symptoms of dwarfism in dogs")
                    st.write("Bones look shorter than normal, Enlarged joints, Abnormal bone shape, Larger head than normal, Shorter nose with an undershot jaw, Crooked teeth, Lack of growth, Spinal deviation (either side), Bowing of forelimbs (leaning out sideways), Heart issues, Fear and dog aggression")
                    st.markdown('---') 
                    st.subheader("Diagnosing dwarfism in dogs")
                    st.write("For a diagnosis of dwarfism, the vet will usually examine your dog and may carry out X-rays to look at bone development. Make note of any more abnormalities you’ve spotted. To rule out other conditions and further confirm if a dog has dwarfism, the vet may want to take blood and send it to the lab for further testing.")
                    st.markdown("---")
                    st.subheader("Can dwarfism in dogs be treated?")
                    st.write("Though there is no cure for dwarfism, whatever the cause, your vet may suggest medication to help manage the condition together with pain relief if required. For example, it is possible to treat and manage pituitary dwarfism in dogs with prescribed dosages of growth hormone. In mature dogs, spaying or neutering could also help. It’s important to talk to a vet first if you suspect that your dog has dwarfism.")
                    st.markdown("---")
                    st.subheader("How long do dogs live with dwarfism? ")
                    st.write("Depending on the severity of the case, some dogs with dwarfism can live relatively normal lives. Other dogs, however, unfortunately, won’t live past 5 years of age. This does depend on the breed and type of dwarfism – your vet will be able to provide more specifics for your dog.")
                    st.markdown("---")
                    st.subheader("Caring for dogs with dwarfism")
                    st.write("Giving your dog a good quality of life is highly important. Dwarfism in dogs can cause certain health issues that will need to be managed to ensure your pooch is as happy as possible. One thing to bear in mind with achondroplasia is that affected dogs tend to be prone to arthritis. Another thing to note is obesityas carrying extra weight puts additional strain on arthritic joints– making sure your dog has a healthy diet and a good amount of exercise is important here. Some pain medication can help too but make sure a vet is consulted for the best course of action.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Acute moist dermatitis")
                    st.image("https://whitehorsevet.com.au/wp-content/uploads/2018/02/dog-hot-spot.jpg")
                with col2:
                    Acute_moist_dermatitis = '''Also referred to as hot spots or pyotraumatic dermatitis, is a skin condition characterized by localized, moist, erythematous areas. It is one of the most common presenting signs associated with canine skin disorders. Clinically the lesions appear to arise secondary to self-induced trauma'''
                    st.markdown(Acute_moist_dermatitis)
                with st.expander("See More Details"):
                    st.subheader("Diagnosis and Prognosis")
                    st.write("Differential diagnosis - Hot spots are rarely confused with other disorders. Determining the underlying cause, however, can be difficult. Potential underlying causes include: Flea allergy, Foreign bodies, Atopy, Food allergy, Otitis externa, Contact irritants, Scabies, Demodex, Anal sac disease, Irritation after clipping or grooming, Dermatophytosis (rare), Drug reaction (rare), Immune mediated disease (rare), Vasculitis (rare).")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("There are 4 key treatment principles. These include: Cleaning and drying the lesion, Systemic anti-inflammatory treatment to stop the itch-scratch cycle, Systemic antibiotics if pyotraumatic folliculitis is present, Identification and control of the underlying disease in the case of recurrent hot spots")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Bloat")
                    st.image("https://www.akc.org/wp-content/uploads/2021/09/Senior-Beagle-lying-on-a-rug-indoors.jpg")
                with col2:
                    Bloat = '''Bloat, also known as gastric dilatation-volvulus (GDV) complex, is a medical and surgical emergency. As the stomach fills with air, pressure builds, stopping blood from the hind legs and abdomen from returning to the heart. Blood pools at the back end of the body, reducing the working blood volume and sending the dog into shock.'''
                    st.markdown(Bloat)
                with st.expander("See More Details"):
                    st.subheader("What Are the Signs of Bloat in Dogs?")
                    st.write("An enlargement of the dog’s abdomen")
                    st.write("Retching")
                    st.write("Salivation")
                    st.write("Restlessness")
                    st.write("An affected dog will feel pain and might whine if you press on his belly")
                    st.write("Without treatment, in only an hour or two, your dog will likely go into shock. The heart rate will rise and the pulse will get weaker, leading to death.")
                    st.markdown("---")
                    st.subheader("Why Do Dogs Bloat?")
                    st.write("This question has perplexed veterinarians since they first identified the disease. We know air accumulates in the stomach (dilatation), and the stomach twists (the volvulus part). We don’t know if the air builds up and causes the twist, or if the stomach twists and then the air builds up.")
                    st.markdown("---")
                    st.subheader("How Is Bloat Treated?")
                    st.write("Veterinarians start by treating the shock. Once the dog is stable, he’s taken into surgery. We do two procedures. One is to deflate the stomach and turn it back to its correct position. If the stomach wall is damaged, that piece is removed. Second, because up to 90 percent of affected dogs will have this condition again, we tack the stomach to the abdominal wall (a procedure called a gastropexy) to prevent it from twisting.")
                    st.markdown("---")
                    st.subheader("How Can Bloat Be Prevented?")
                    st.write("If a dog has relatives (parents, siblings, or offspring) who have suffered from bloat, there is a higher chance he will develop bloat. These dogs should not be used for breeding.")
                    st.write("Risk of bloat is correlated to chest conformation. Dogs with a deep, narrow chest — very tall, rather than wide — suffer the most often from bloat. Great Danes, who have a high height-to-width ratio, are five-to-eight times more likely to bloat than dogs with a low height-to-width ratio.")
                    st.markdown("---")

        elif breed_label == "Beagle":
            tab1, tab2, tab3= st.tabs(["Atopic dermatitis", "Cataract", "Epilepsy"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Atopic dermatitis")
                    st.image("https://images.ctfassets.net/82d3r48zq721/6jiyt1463nRXCZa451La8T/5114391b07c7760c8b9068224869ae42/Dog-with-skin-allergy-atopic-dematitis-biting-and-scratching-himself_resized.jpg?w=800&h=531&q=50&fm=webp")
                with col2:
                    Atopic_dermatitis = '''Atopic dermatitis (or atopy) is an inflammatory, chronic skin condition associated with environmental allergies and is the second most common allergic skin condition diagnosed in dogs.'''
                    st.markdown(Atopic_dermatitis)
                with st.expander("See More Details"):
                    st.subheader("Signs & symptoms of atopic dermatitis in dogs")
                    st.write("A dog with atopic dermatitis will usually show signs and symptoms between 3 months to 6 years of age. It’s not as common for dogs over the age of 7 to develop atopic dermatitis, although a new environment can trigger new allergens. Atopic dermatitis often begins as a mild condition with symptoms not becoming clinically visible before the third year.")
                    st.markdown("---")
                    st.subheader("What causes atopic dermatitis in dogs?")
                    st.write("Atopic dermatitis is a genetic disease that is predisposed in some breeds more than others. For that reason, dogs diagnosed with the condition should not be bred. The cause is unknown, but a general understanding of the anatomy of the skin is vital in understanding what happens to a dog when the skin becomes irritated and inflamed as a result of allergens in the environment. A case of atopic dermatitis can be painful and uncomfortable for a dog.")
                    st.markdown("---")
                    st.subheader("Diagnosing canine atopic dermatitis")
                    st.write("Symptoms of atopic dermatitis are similar to other skin conditions, which can make it difficult to diagnose. Uncovering the cause may take time and is often a process of elimination. Along with a full medical examination, which includes a look at the dog’s complete medical history, additional allergy testing may be done. In some cases, your veterinarian may perform a blood test (serological allergy test) to determine the presence of an antibody called IgE to specific allergens. An increase in an allergen-specific IgE usually means there is an overreaction to that allergen in the body.")
                    st.markdown("---")
                    st.subheader("Treatment for atopic dermatitis in dogs")
                    st.write("One of the first steps is eliminating or reducing exposure to the allergens causing dermatitis. If you are unable to identify the irritants, use a process of elimination by removing the environmental factors that have the potential to trigger an outbreak. Diet, bedding, even the general environment in which the dog is exposed to may need to be changed.")
                    st.write("For dogs with a severe case of atopic dermatitis, removing and changing specific factors might not be enough. Oral corticosteroids can be given to control or reduce the itching and swelling, but there are side effects associated with steroids, so it’s important to administer as directed by your veterinarian. There are also other non-steroidal drugs that your veterinarian might prescribe to alleviate the discomfort.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
            with tab3: 
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Epilepsy")
                    st.image("https://canna-pet.com/wp-content/uploads/2017/03/CP_EpilepsyDogs_1.jpg")
                with col2:
                    Epilepsy = '''The most common neurological disorder seen in dogs, and has been estimated to affect approximately 0.75% of the canine population3. The term epilepsy refers to a heterogeneous disease that is characterized by the presence of recurrent, unprovoked seizures resulting from an abnormality of the brain. '''
                    st.markdown(Epilepsy)
                with st.expander("See More Details"):
                    st.subheader("What Are the Symptoms of Seizures?")
                    st.write("Symptoms can include collapsing, jerking, stiffening, muscle twitching, loss of consciousness, drooling, chomping, tongue chewing, or foaming at the mouth. Dogs can fall to the side and make paddling motions with their legs. They sometimes poop or pee during the seizure. They are also not aware of their surroundings. Some dogs may look dazed, seem unsteady or confused, or stare off into space before a seizure. Afterward, your dog may be disoriented, wobbly, or temporarily blind. They may walk in circles and bump into things. They might have a lot of drool on their chin. They may try to hide.")
                    st.markdown("---")
                    st.subheader("What Should I Do if My Dog Has a Seizure?")
                    st.write("First, try to stay calm. If your dog is near something that could hurt them, like a piece of furniture or the stairs, gently slide them away.")
                    st.write("Stay away from your dog’s mouth and head; they could bite you. Don’t put anything in their mouth. Dogs cannot choke on their tongues. If you can, time it.")
                    st.write("If the seizure lasts for more than a couple of minutes, your dog is at risk of overheating. Turn a fan on your dog and put cold water on their paws to cool them down.")
                    st.write("Talk to your dog softly to reassure them. Avoid touching them - they may unknowingly bite. Call your vet when the seizure ends.")
                    st.write("If dogs have a seizure that lasts more than 5 minutes or  have several in a row while they are unconscious, take them to a vet as soon as possible. The longer a seizure goes on, the higher a dog’s body temperature can rise, and they may have problems breathing. This can raise their risk of brain damage. Your vet may give your dog IV Valium to stop the seizure.")
                    st.markdown("---")
                    st.subheader("What Should I Expect When I Take My Dog to the Vet?")
                    st.write("Your vet will want to do a thorough physical exam and get some lab work to look for the causes of your dog’s seizures. Diagnostic imaging like MRI can help detect brain lesions. ")
                    st.write("Your vet may prescribe medicines to control seizures. Always follow your vet’s instructions when you give your dog medicine. Don’t let them miss a dose.")
                    st.markdown("---")

        elif breed_label == "Bloodhound":
            tab1, tab2, tab3= st.tabs(["Bloat", "Entropion", "Gastric torsion"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Bloat")
                    st.image("https://www.akc.org/wp-content/uploads/2021/09/Senior-Beagle-lying-on-a-rug-indoors.jpg")
                with col2:
                    Bloat = '''Bloat, also known as gastric dilatation-volvulus (GDV) complex, is a medical and surgical emergency. As the stomach fills with air, pressure builds, stopping blood from the hind legs and abdomen from returning to the heart. Blood pools at the back end of the body, reducing the working blood volume and sending the dog into shock.'''
                    st.markdown(Bloat)
                with st.expander("See More Details"):
                    st.subheader("What Are the Signs of Bloat in Dogs?")
                    st.write("An enlargement of the dog’s abdomen")
                    st.write("Retching")
                    st.write("Salivation")
                    st.write("Restlessness")
                    st.write("An affected dog will feel pain and might whine if you press on his belly")
                    st.write("Without treatment, in only an hour or two, your dog will likely go into shock. The heart rate will rise and the pulse will get weaker, leading to death.")
                    st.markdown("---")
                    st.subheader("Why Do Dogs Bloat?")
                    st.write("This question has perplexed veterinarians since they first identified the disease. We know air accumulates in the stomach (dilatation), and the stomach twists (the volvulus part). We don’t know if the air builds up and causes the twist, or if the stomach twists and then the air builds up.")
                    st.markdown("---")
                    st.subheader("How Is Bloat Treated?")
                    st.write("Veterinarians start by treating the shock. Once the dog is stable, he’s taken into surgery. We do two procedures. One is to deflate the stomach and turn it back to its correct position. If the stomach wall is damaged, that piece is removed. Second, because up to 90 percent of affected dogs will have this condition again, we tack the stomach to the abdominal wall (a procedure called a gastropexy) to prevent it from twisting.")
                    st.markdown("---")
                    st.subheader("How Can Bloat Be Prevented?")
                    st.write("If a dog has relatives (parents, siblings, or offspring) who have suffered from bloat, there is a higher chance he will develop bloat. These dogs should not be used for breeding.")
                    st.write("Risk of bloat is correlated to chest conformation. Dogs with a deep, narrow chest — very tall, rather than wide — suffer the most often from bloat. Great Danes, who have a high height-to-width ratio, are five-to-eight times more likely to bloat than dogs with a low height-to-width ratio.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Entropion")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Entroption = '''when the eyelids roll inward toward the eye. The fur on the eyelids and the eyelashes then rub against the surface of the eye (the cornea). This is a very painful condition that can lead to corneal ulcers.'''
                    st.markdown(Entroption)
                with st.expander("See More Details"):
                    st.write("Many Bloodhounds have abnormally large eyelids (macroblepharon) which results in an unusually large space between the eyelids.  Because of their excessive facial skin and resulting facial droop, there is commonly poor support of the outer corner of the eyelids")
                    st.markdown("---")
                    st.subheader("How is entropion treated?")
                    st.write("The treatment for entropion is surgical correction. A section of skin is removed from the affected eyelid to reverse its inward rolling. In many cases, a primary, major surgical correction will be performed, and will be followed by a second, minor corrective surgery later. Two surgeries are often performed to reduce the risk of over-correcting the entropion, resulting in an outward-rolling eyelid known as ectropion. Most dogs will not undergo surgery until they have reached their adult size at six to twelve months of age.")
                    st.markdown("---")
                    st.subheader("Should an affected dog be bred?")
                    st.write("Due to the concern of this condition being inherited, dogs with severe ectropion requiring surgical correction should not be bred.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Gastric torsion")
                    st.image("https://www.mcahonline.com/wp-content/uploads/2020/06/MCAH_6.22.2020-300x200.jpg")
                with col2:
                    Gastric_torsion = '''Also known as bloat, a twisted stomach, or Gastric Dilation-Volvulus (GDV). There are two parts to this condition. The first part is the bloating where a dog’s stomach fills up with gas, fluid, food, or any combination of the three massively. Torsion or volvulus is the second part where the entire stomach twists around itself inside of the abdomen. As a result, the abdomen closes off at both the entrance and exit. Today we’re going to go over the causes of gastric torsion, the signs/symptoms, and treatment. '''
                    st.markdown(Gastric_torsion)
                with st.expander("See More Details"):
                    st.subheader("What causes gastric torsion?")
                    st.write("Veterinarians aren’t sure about the exact cause of bloat or torsion. However, we believe some factors can put dogs at a higher risk. These factors include:")
                    st.write("Your pet eating from a food bowl that’s too high")
                    st.write("A dog eating only one big meal a day")
                    st.write("A dog eating too quickly")
                    st.write("A dog running or playing right after they eat")
                    st.write("Genetics")
                    st.write("Overeating/drinking")
                    st.write("If your dog is a part of a bigger breed, esp. if they have deep chests.")
                    st.write("If your dog is older (especially older than seven years old)")
                    st.write("Male dogs also experience gastric torsion more often than female dogs")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("If your canine companion shows all of the classic signs of gastric torsion, this is a life-threatening condition and you need to bring your pet to a veterinarian immediately. One of our experienced veterinarians will evaluate your dog and take an x-ray for confirmation. If your dog is experiencing shock, we will place an IV catheter to administer fluids and medications to stabilize your pet prior to surgery. ")
                    st.write("Treatments vary depending on the severity of the condition. If your pet is not experiencing torsion, a veterinarian can put a tube down the throat to release any built-up pressure. A twisted stomach (determined via X-ray) can stop the tube from entering the throat. In the case of a twisted stomach, emergency surgery will need to happen. Aside from surgery, your dog will need continued fluids through an IV and medications. We will also continue to monitor your dog’s heart for any signs of abnormalities that can be a side effect of gastric torsion.")
                    st.write("The good news is that as with any condition, the earlier you detect the signs, the better. Prevention involves making sure that your fur baby is eating at eye level, not playing right away after mealtime, and making sure that they’re eating well-balanced meals (2-3 small meals a day). For those high-risk breeds, we can also perform a surgery called a gastropexy. This is when one of our veterinarians tacks the stomach to the body wall, greatly reducing the likelihood that it can twist. If you have more questions about gastric torsion or gastropexy, call to talk to one of our amazing veterinarians!")
                    st.markdown("---")

        elif breed_label == "Bluetick":
            tab1, tab2, tab3= st.tabs(["Osteochondritis dissecans", "Globoid cell leukodystrophy", "Lysosomal ‘storage’ diseases"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Osteochondritis dissecans")
                    st.image("https://ntp.niehs.nih.gov/sites/default/files/nnl/musculoskeletal/bone/osteoch/images/figure-001-a75875_large.jpg")
                with col2:
                    Osteochondrosis = '''A specific form of inflammation of the cartilage of certain joints which causes arthritis'''
                    st.markdown(Osteochondrosis)
                with st.expander("See More Details"):
                    st.subheader("Symptoms")
                    st.write("Lameness (most common symptom), Onset of lameness may be sudden or gradual, and may involve one or more limbs, Lameness becomes worse after exercise, Unable to bear weight on affected limb, Swelling at joints, Pain in limb, especially on manipulation of joints involved, Wasting of muscles with chronic lameness")
                    st.markdown("---")
                    st.subheader("Cause")
                    st.write("Unknown, Appears to be genetically acquired, Disruption in supply of blood to the bone or through the bone, Nutritional deficiencies")
                    st.markdown("---")
                    st.subheader("Diagnose")
                    st.write("You will need to give a thorough medical history of your dog's health, onset of symptoms, and any information you have about your dog's parentage. A complete blood profile will be conducted, including a chemical blood profile, a complete blood count, and a urinalysis. The results of these tests are often within normal ranges in affected animals, but they are necessary for preliminary assumptions of your dog's overall health condition. Your veterinarian will examine your dog thoroughly, paying special attention to the limbs that are troubling your dog. Radiography imaging is the best tool for diagnosis of this problem; your veterinarian will take several x-rays of the affected joints and bones to best discern any abnormalities. The radiographs may show details of lesions and abnormalities related to this disease. Computed tomography (CT-scan) and magnetic resonance imaging (MRI) are also valuable diagnostic tools for visualizing the extent of any internal lesions. Your veterinarian will also take samples of fluid from the affected joints (synovial fluid) to confirm involvement of the joint and to rule out an infectious disease that may be the actual cause of the lameness. More advanced diagnostic and therapeutic tools like arthroscopy may also be used. Arthroscopy is a minimally invasive surgical procedure which allows for examination and sometime treatment of damage inside the joint. This procedure is performed using an arthroscope, a type of endoscope inserted into the joint through a small incision")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write(" After establishing the diagnosis, your veterinarian will plan corrective surgery. Either arthroscopy or arthrotomy (surgical incision into the joint) techniques can be used to reach the area. Your veterinarian will presribe medicines to control pain and inflammation for a few days after surgery. There are also some medicines that are available, and that are known to limit the cartilage damage and degeneration. Your doctor will explain your options to you based on the final diagnosis.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Globoid cell leukodystrophy")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    globoid = '''Krabbe Disease, also known as Globoid Cell Leukodystrophy (abbreviated as GCL), is a severe neurological disorder that affects humans and dogs. Affected puppies have difficulty walking, tremors, and loss of muscle mass starting as early as 1-3 months of age.'''
                    st.markdown(globoid)
                with st.expander("See More Details"):
                    st.subheader("When do clinical signs appear?")
                    st.write("Young puppies do not generally show signs of GCL. When signs emerge, typically between one and six months in small terriers, the puppy will begin having tremors and overall muscle weakness and will seem to lose control over his legs. They stop growing normally, and they may develop vision problems. Larger breed dogs with GCL may not show signs until between 18 months and five years of age.")
                    st.write("In addition to the neuromuscular symptoms, affected puppies will develop behavior changes, dementia, lack of appetite, starvation from an inability to eat, and urinary and fecal incontinence. Most dogs must be euthanized due to their neuromuscular decline about two to six months after the onset of clinical signs.")
                    st.markdown("---")
                    st.subheader("Are certain breeds more susceptible to GCL?")
                    st.write("Breeds most commonly affected by GCL include the West Highland White Terrier, Cairn Terrier, Bluetick Coonhound, Beagle, Poodle, Basset Hound, Pomeranian, and Irish Setter.")
                    st.markdown("---")
                    st.subheader("How is this disease diagnosed?")
                    st.write("A definitive diagnosis of GCL can be made with a blood test in Cairn Terriers, West Highland White Terriers, and Irish Setters. GCL is caused by a recessive gene, meaning both parents must be carriers. Even with two recessive carriers, not all litters or all puppies in a particular litter will have GCL. They may, however, be carriers. Reputable breeders work to identify carriers and eliminate them from their breeding program")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Lysosomal Storage Disease")
                    st.image("https://www.akc.org/wp-content/uploads/2017/11/Bluetick-Coonhound.jpg")
                with col2:
                    Lysosomal = '''Inborn errors of metabolism characterized by the accumulation of substrates in excess in various organs' cells due to the defective functioning of lysosomes. They cause dysfunction of those organs where they accumulate and contribute to great morbidity and mortality.'''
                    st.markdown(Lysosomal)
                with st.expander("See More Details"):
                    st.subheader("What causes lysosomal storage diseases?")
                    st.write("Lysosomal storage diseases are rare and are inherited through recessive genes, meaning both parents are carriers. Responsible breeders remove these dogs from their breeding programs.")
                    st.markdown("---")
                    st.subheader("What are the clinical signs of lysosomal storage diseases?")
                    st.write("The clinical signs of lysosomal storage diseases vary depending on the enzyme that is missing, the cells involved, and the material that is accumulating. In general, the most common symptoms seen in puppies with lysosomal storage diseases include failure to thrive, incoordination and balance issues, exercise intolerance, abnormal vision or progression to blindness, fainting, and seizures.")
                    st.markdown("---")
                    st.subheader("Is there any treatment for lysosomal storage diseases?")
                    st.write("Unfortunately, lysosomal storage diseases are uniformly fatal. The disorders manifest shortly after birth, progress rapidly, and the affected dogs typically die between four and six months of age. The missing enzymes are involved in basic metabolic functions, and without them, the body simply cannot function and thrive. These are a devastating group of diseases, and it is fortunate that they are quite rare.")
                    st.markdown("---")
        elif breed_label == "Black and Tan Coonhound":
            tab1, tab2, tab3= st.tabs(["Entropion", "Hip dysplasia", "Osteochondritis dissecans"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Entropion")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Entroption = '''when the eyelids roll inward toward the eye. The fur on the eyelids and the eyelashes then rub against the surface of the eye (the cornea). This is a very painful condition that can lead to corneal ulcers.'''
                    st.markdown(Entroption)
                with st.expander("See More Details"):
                    st.write("Many Bloodhounds have abnormally large eyelids (macroblepharon) which results in an unusually large space between the eyelids.  Because of their excessive facial skin and resulting facial droop, there is commonly poor support of the outer corner of the eyelids")
                    st.markdown("---")
                    st.subheader("How is entropion treated?")
                    st.write("The treatment for entropion is surgical correction. A section of skin is removed from the affected eyelid to reverse its inward rolling. In many cases, a primary, major surgical correction will be performed, and will be followed by a second, minor corrective surgery later. Two surgeries are often performed to reduce the risk of over-correcting the entropion, resulting in an outward-rolling eyelid known as ectropion. Most dogs will not undergo surgery until they have reached their adult size at six to twelve months of age.")
                    st.markdown("---")
                    st.subheader("Should an affected dog be bred?")
                    st.write("Due to the concern of this condition being inherited, dogs with severe ectropion requiring surgical correction should not be bred.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Osteochondritis dissecans")
                    st.image("https://ntp.niehs.nih.gov/sites/default/files/nnl/musculoskeletal/bone/osteoch/images/figure-001-a75875_large.jpg")
                with col2:
                    Osteochondrosis = '''A specific form of inflammation of the cartilage of certain joints which causes arthritis'''
                    st.markdown(Osteochondrosis)
                with st.expander("See More Details"):
                    st.subheader("Symptoms")
                    st.write("Lameness (most common symptom), Onset of lameness may be sudden or gradual, and may involve one or more limbs, Lameness becomes worse after exercise, Unable to bear weight on affected limb, Swelling at joints, Pain in limb, especially on manipulation of joints involved, Wasting of muscles with chronic lameness")
                    st.markdown("---")
                    st.subheader("Cause")
                    st.write("Unknown, Appears to be genetically acquired, Disruption in supply of blood to the bone or through the bone, Nutritional deficiencies")
                    st.markdown("---")
                    st.subheader("Diagnose")
                    st.write("You will need to give a thorough medical history of your dog's health, onset of symptoms, and any information you have about your dog's parentage. A complete blood profile will be conducted, including a chemical blood profile, a complete blood count, and a urinalysis. The results of these tests are often within normal ranges in affected animals, but they are necessary for preliminary assumptions of your dog's overall health condition. Your veterinarian will examine your dog thoroughly, paying special attention to the limbs that are troubling your dog. Radiography imaging is the best tool for diagnosis of this problem; your veterinarian will take several x-rays of the affected joints and bones to best discern any abnormalities. The radiographs may show details of lesions and abnormalities related to this disease. Computed tomography (CT-scan) and magnetic resonance imaging (MRI) are also valuable diagnostic tools for visualizing the extent of any internal lesions. Your veterinarian will also take samples of fluid from the affected joints (synovial fluid) to confirm involvement of the joint and to rule out an infectious disease that may be the actual cause of the lameness. More advanced diagnostic and therapeutic tools like arthroscopy may also be used. Arthroscopy is a minimally invasive surgical procedure which allows for examination and sometime treatment of damage inside the joint. This procedure is performed using an arthroscope, a type of endoscope inserted into the joint through a small incision")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write(" After establishing the diagnosis, your veterinarian will plan corrective surgery. Either arthroscopy or arthrotomy (surgical incision into the joint) techniques can be used to reach the area. Your veterinarian will presribe medicines to control pain and inflammation for a few days after surgery. There are also some medicines that are available, and that are known to limit the cartilage damage and degeneration. Your doctor will explain your options to you based on the final diagnosis.")
                    st.markdown("---")
        elif breed_label == "Walker Hound":
            tab1, tab2, tab3= st.tabs(["Hip Dysplasia", "Cataract", "Ear Infections"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Ear Infection")
                    st.image("https://www.akc.org/wp-content/uploads/2018/08/german-shepherd-ears-examined-by-vet.jpg")
                with col2:
                    Ear = '''Ear infections are common conditions in dogs, especially those with floppy ears such as Basset Hounds and Cocker Spaniels. An estimated 20 percent of dogs have some form of ear disease, which may affect one or both ears. Fortunately, there are steps you can take to reduce the length and severity of these episodes for your dog. There are dog ear care products to help prevent issues from reoccurring.'''
                    st.markdown(Ear)
                with st.expander("See More Details"):
                    st.subheader("Symptoms of Dog Ear Infections")
                    st.write("Some dogs show no symptoms of ear infection aside from a buildup of wax and discharge in the ear canal. But ear infections often cause significant discomfort and affected dogs may show signs such as:")
                    st.write("Head shaking")
                    st.write("Scratching at the affected ear")
                    st.write("Dark discharge")
                    st.write("Odor")
                    st.write("Redness and swelling of the ear canal")
                    st.write("Pain")
                    st.write("Itchiness")
                    st.write("Crusting or scabs in the ears")
                    st.markdown("---")
                    st.subheader("What Causes Ear Infections in Dogs?")
                    st.write("The canine ear canal is more vertical than that of a human, forming an L-shape that tends to hold in fluid. This makes dogs more prone to ear infections. Ear infections are typically caused by bacteria, yeast, or a combination of both. In puppies, ear mites can also be a source of infection")
                    st.markdown("---")
                    st.subheader("Precise Diagnosis Needed for a Dog’s Ear Infections")
                    st.write("If your dog is showing any of the common signs of ear infections, it’s important to consult with your veterinarian as soon as possible. Quick treatment is necessary not only for your dog’s comfort (these conditions can be painful!), but also to prevent the spread of infection to the middle and inner ear. Don’t try to treat ear infections at home.")
                    st.write("Be prepared to provide your vet with a thorough history of the problem. This is especially important for first-time infections, or if you are seeing a new veterinarian.")
                    st.markdown("---")
                    st.subheader("How are Dog Ear Infections Treated?")
                    st.write("Your veterinarian will thoroughly clean your dog’s ears using a medicated ear cleanser. Your vet may also prescribe your dog prescription ear drops for you to use at home. In severe cases of dog ear infections, your vet may prescribe oral antibiotics and anti-inflammatory medications.")
                    st.write("Most uncomplicated ear infections resolve within 1–2 weeks, once appropriate treatment begins. But severe infections or those due to underlying conditions may take months to resolve, or may become chronic problems. In cases of severe chronic disease where other treatments have failed, your veterinarian may recommend surgery such as a Total Ear Canal Ablation (TECA). A TECA surgery removes the ear canal, thus removing the diseased tissue and preventing the recurrence of infection.")
                    st.write("It is important to follow your veterinarian’s instructions closely and return to the veterinary hospital for any recommended recheck appointments. Lapses in your dog’s treatment may lead to the recurrence of the infection. It is especially important that you finish the full course of your dog’s medication, even if your dog appears to be getting better. Failure to finish the full course of treatment may lead to additional problems such as resistant infections.")
                    st.markdown("---")

        elif breed_label == "English Foxhound":
            tab1, tab2, tab3= st.tabs(["Deafness", "Hip dysplasia", "Renal dysplasia"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Deafness")
                    st.image("https://www.aaha.org/contentassets/a9895c6d4d55453a8080abd33a77e2e6/blog-2-test---gettyimages-1330594294.png")
                with col2:
                    Deafness = '''An inability to hear, due to many different causes. In Dalmatians, congenital deafness is associated with blue eye color. Deafness may be congenital (present at birth) or acquired as a result of infection, trauma, or degeneration of the cochlea (the organ of hearing).'''
                    st.markdown(Deafness)
                with st.expander("See More Details"):
                    st.write("Deafness present at birth can be inherited or result from toxic or viral damage to the developing unborn puppy. Merle and white coat colors are associated with deafness at birth in dogs and other animals. Dog breeds commonly affected include the Dalmatian, Bull Terrier, Australian Heeler, Catahoula, English Cocker Spaniel, Parson Russell Terrier, and Boston Terrier. The list of affected breeds (now approximately 100) continues to expand and may change due to breed popularity and elimination of the defect through selective breeding.")
                    st.markdown("---")
                    st.write("Acquired deafness may result from blockage of the external ear canal due to longterm inflammation (otitis externa) or excessive ear wax. It may also occur due to a ruptured ear drum or inflammation of the middle or inner ear. Hearing usually returns after these types of conditions are resolved.")
                    st.markdown("---")
                    st.write("The primary sign of deafness is failure to respond to a sound, for example, failure of noise to awaken a sleeping dog, or failure to alert to the source of a sound. Other signs include unusual behavior such as excessive barking, unusual voice, hyperactivity, confusion when given vocal commands, and lack of ear movement. An animal that has gradually become deaf, as in old age, may become unresponsive to the surroundings and refuse to answer the owner’s call.")
                    st.markdown("---")
                    st.write("Deaf dogs do not appear to experience pain or discomfort due to the condition. However, caring for a dog that is deaf in both ears requires more dedication than owning a hearing dog. These dogs are more likely to be startled, which can lead to biting. These dogs are also less protected from certain dangers, such as motor vehicles.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Renal dysplasia")
                    st.image("https://avropethospital.com/files/2011/05/RenalDysplasia.jpg")
                with col2:
                    Renal_dysplasia = '''A kidney does not fully develop in the womb. The affected kidney does not have normal function – which means that it does not work as well as a normal kidney. It is usually smaller than usual, and may have some cysts, which are like sacs filled with liquid. '''
                    st.markdown(Renal_dysplasia)
                with st.expander("See More Details"):
                    st.write("It is one type of congenital anomaly of the kidneys and urinary tract. ‘Congenital’ means that the problem is present at birth and ‘anomaly’ means different than normal.")
                    st.markdown("---")
                    st.subheader("How common is it?")
                    st.write("Renal dysplasia is relatively common. It is estimated that one baby in a few hundred may be affected. ")
                    st.write("Renal dysplasia may be picked up before birth on the 20 week antenatal ultrasound scan, or soon after birth. It may also be picked up in an older child who has some symptoms. An affected kidney is called a dysplastic kidney. Renal dysplasia rarely causes any problems during the pregnancy or in childbirth.")
                    st.subheader("Causes")
                    st.write("Renal dysplasia happens when part of the kidney does not develop properly in the womb. It is relatively common. It is not always possible to know why renal dysplasia happens. In the majority of cases, it is not caused by anything that the mother does during her pregnancy, and it is unlikely that a future pregnancy will result in renal dysplasia or other problems with the kidneys. Doctors understand that there are some possible causes of renal dysplasia, though it may not always be possible to identify the cause in your baby. It is not usually caused by anything that the mother does during her pregnancy. Occasionally a specific cause is found.")
                    st.subheader("Diagnosis later in childhood")
                    st.write("Sometimes, renal dysplasia is only picked up after birth or when a child is older. It is usually found during a scan that a child is having for another reason, such as after a urinary tract infection (UTI) or after an accident.")
                    st.markdown("---")
            
        elif breed_label == "Redbone":
            tab1, tab2, tab3= st.tabs(["Ear Infections", "Hip dysplasia", "Cataract"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Ear Infection")
                    st.image("https://www.akc.org/wp-content/uploads/2018/08/german-shepherd-ears-examined-by-vet.jpg")
                with col2:
                    Ear = '''Ear infections are common conditions in dogs, especially those with floppy ears such as Basset Hounds and Cocker Spaniels. An estimated 20 percent of dogs have some form of ear disease, which may affect one or both ears. Fortunately, there are steps you can take to reduce the length and severity of these episodes for your dog. There are dog ear care products to help prevent issues from reoccurring.'''
                    st.markdown(Ear)
                with st.expander("See More Details"):
                    st.subheader("Symptoms of Dog Ear Infections")
                    st.write("Some dogs show no symptoms of ear infection aside from a buildup of wax and discharge in the ear canal. But ear infections often cause significant discomfort and affected dogs may show signs such as:")
                    st.write("Head shaking")
                    st.write("Scratching at the affected ear")
                    st.write("Dark discharge")
                    st.write("Odor")
                    st.write("Redness and swelling of the ear canal")
                    st.write("Pain")
                    st.write("Itchiness")
                    st.write("Crusting or scabs in the ears")
                    st.markdown("---")
                    st.subheader("What Causes Ear Infections in Dogs?")
                    st.write("The canine ear canal is more vertical than that of a human, forming an L-shape that tends to hold in fluid. This makes dogs more prone to ear infections. Ear infections are typically caused by bacteria, yeast, or a combination of both. In puppies, ear mites can also be a source of infection")
                    st.markdown("---")
                    st.subheader("Precise Diagnosis Needed for a Dog’s Ear Infections")
                    st.write("If your dog is showing any of the common signs of ear infections, it’s important to consult with your veterinarian as soon as possible. Quick treatment is necessary not only for your dog’s comfort (these conditions can be painful!), but also to prevent the spread of infection to the middle and inner ear. Don’t try to treat ear infections at home.")
                    st.write("Be prepared to provide your vet with a thorough history of the problem. This is especially important for first-time infections, or if you are seeing a new veterinarian.")
                    st.markdown("---")
                    st.subheader("How are Dog Ear Infections Treated?")
                    st.write("Your veterinarian will thoroughly clean your dog’s ears using a medicated ear cleanser. Your vet may also prescribe your dog prescription ear drops for you to use at home. In severe cases of dog ear infections, your vet may prescribe oral antibiotics and anti-inflammatory medications.")
                    st.write("Most uncomplicated ear infections resolve within 1–2 weeks, once appropriate treatment begins. But severe infections or those due to underlying conditions may take months to resolve, or may become chronic problems. In cases of severe chronic disease where other treatments have failed, your veterinarian may recommend surgery such as a Total Ear Canal Ablation (TECA). A TECA surgery removes the ear canal, thus removing the diseased tissue and preventing the recurrence of infection.")
                    st.write("It is important to follow your veterinarian’s instructions closely and return to the veterinary hospital for any recommended recheck appointments. Lapses in your dog’s treatment may lead to the recurrence of the infection. It is especially important that you finish the full course of your dog’s medication, even if your dog appears to be getting better. Failure to finish the full course of treatment may lead to additional problems such as resistant infections.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
        elif breed_label == "Borzoi":
            tab1, tab2, tab3= st.tabs(["Bloat", "Hip dysplasia", "Hygroma"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Bloat")
                    st.image("https://www.akc.org/wp-content/uploads/2021/09/Senior-Beagle-lying-on-a-rug-indoors.jpg")
                with col2:
                    Bloat = '''Bloat, also known as gastric dilatation-volvulus (GDV) complex, is a medical and surgical emergency. As the stomach fills with air, pressure builds, stopping blood from the hind legs and abdomen from returning to the heart. Blood pools at the back end of the body, reducing the working blood volume and sending the dog into shock.'''
                    st.markdown(Bloat)
                with st.expander("See More Details"):
                    st.subheader("What Are the Signs of Bloat in Dogs?")
                    st.write("An enlargement of the dog’s abdomen")
                    st.write("Retching")
                    st.write("Salivation")
                    st.write("Restlessness")
                    st.write("An affected dog will feel pain and might whine if you press on his belly")
                    st.write("Without treatment, in only an hour or two, your dog will likely go into shock. The heart rate will rise and the pulse will get weaker, leading to death.")
                    st.markdown("---")
                    st.subheader("Why Do Dogs Bloat?")
                    st.write("This question has perplexed veterinarians since they first identified the disease. We know air accumulates in the stomach (dilatation), and the stomach twists (the volvulus part). We don’t know if the air builds up and causes the twist, or if the stomach twists and then the air builds up.")
                    st.markdown("---")
                    st.subheader("How Is Bloat Treated?")
                    st.write("Veterinarians start by treating the shock. Once the dog is stable, he’s taken into surgery. We do two procedures. One is to deflate the stomach and turn it back to its correct position. If the stomach wall is damaged, that piece is removed. Second, because up to 90 percent of affected dogs will have this condition again, we tack the stomach to the abdominal wall (a procedure called a gastropexy) to prevent it from twisting.")
                    st.markdown("---")
                    st.subheader("How Can Bloat Be Prevented?")
                    st.write("If a dog has relatives (parents, siblings, or offspring) who have suffered from bloat, there is a higher chance he will develop bloat. These dogs should not be used for breeding.")
                    st.write("Risk of bloat is correlated to chest conformation. Dogs with a deep, narrow chest — very tall, rather than wide — suffer the most often from bloat. Great Danes, who have a high height-to-width ratio, are five-to-eight times more likely to bloat than dogs with a low height-to-width ratio.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/bloat-in-dogs/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hip-dysplasia-in-dogs/")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hygroma")
                    st.image("https://img-va.myshopline.com/image/store/1668653400963/ProP-Bandages-Fig-1.png?w=728&h=437")
                with col2:
                    Hygroma = '''A hygroma is a fluid-filled swelling surrounded by a thick capsule of fibrous tissue that develops under the skin. Hygromas are typically not painful. They can form over any bony prominence on the dog’s body, such as the side of the hock (ankle) joint or over the side of the hip, but they are most commonly found over the elbow. When they first form, hygromas are usually small, soft, and fairly mobile. They may never grow large enough to even notice. If they do grow larger, they can become unsightly or hard to the touch. In the worst case scenario, hygromas can become infected, in which case they are painful and must be dealt with aggressively.'''
                    st.markdown(Hygroma)
                with st.expander("See More Details"):
                    st.subheader("Is a hygroma a tumor? What caused it?")
                    st.write("Hygromas are not tumors. They occur in response to repeated trauma to the tissue over a bony prominence. That is one reason why the elbow is the most common site for hygromas to develop. Especially for large and giant breed dogs, the repeated trauma of lying down on hard surfaces — hardwood, tile, or concrete floors — can produce an inflammatory response in the tissue under the skin over the elbow. The body tries to protect the inflamed area by creating the equivalent of a 'pillow' If the trauma continues, the hygroma will grow larger.")
                    st.write("Hygromas tend to be more common in dogs that are sedentary and spend more of their time lying down, thus increasing the time that pressure is applied over the hygroma site. That said, any large or giant breed dog of any age is potentially at risk of developing a hygroma if they spend time resting and sleeping on hard surfaces.")
                    st.markdown("---")
                    st.subheader("How are hygromas treated?")
                    st.write("The first step in treating a hygroma is to prevent further trauma by providing bedding with adequate padding. Egg-shell foam or memory foam beds may provide the best padding. In areas where the dog enjoys relaxing, the floor can be covered with interlocking foam tiles like the ones found in fitness clubs and day cares. Padded surfaces alone may be all that is required for stabilizing the hygroma.")
                    st.write("There are now commercially available elbow pads designed specifically to protect hygromas (dogleggs™), in order to prevent them from growing or becoming infected and painful. These elbow pads are sized to the dog and then adjustable for a fine-tuned fit. Most dogs will easily tolerate protective elbow pads.")
                    st.write("Should the hygroma grow to an unwieldy size or become infected, it will need to be treated with appropriate antibiotic therapy and may need to be removed surgically. Surgery does not address the underlying cause of the hygroma, so protecting the involved area post-op will be critical for good healing. Your veterinarian will help you to decide how best to proceed with your dog’s hygroma.")
                    st.markdown("---")
                    st.subheader("Can hygromas be prevented?")
                    st.write("There are several things to consider for preventing hygromas. The most important is to ensure that a large or giant breed dog is not allowed to become overweight or obese. Extra weight greatly increases the risk for trauma to the tissue over bony prominences. In addition, it is important to provide bedding with adequate padding. Finally, it is worth covering hard floor surfaces in areas where the dog likes to relax.")
                    st.write("Once a dog has developed a hygroma, attention to a few details can prevent this condition from becoming a problem.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hygroma-in-dogs")
        elif breed_label == "Irish Wolfhound":
            tab1, tab2, tab3= st.tabs(["Bloat", "Cardiomyopathy", "Hygroma"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Bloat")
                    st.image("https://www.akc.org/wp-content/uploads/2021/09/Senior-Beagle-lying-on-a-rug-indoors.jpg")
                with col2:
                    Bloat = '''Bloat, also known as gastric dilatation-volvulus (GDV) complex, is a medical and surgical emergency. As the stomach fills with air, pressure builds, stopping blood from the hind legs and abdomen from returning to the heart. Blood pools at the back end of the body, reducing the working blood volume and sending the dog into shock.'''
                    st.markdown(Bloat)
                with st.expander("See More Details"):
                    st.subheader("What Are the Signs of Bloat in Dogs?")
                    st.write("An enlargement of the dog’s abdomen")
                    st.write("Retching")
                    st.write("Salivation")
                    st.write("Restlessness")
                    st.write("An affected dog will feel pain and might whine if you press on his belly")
                    st.write("Without treatment, in only an hour or two, your dog will likely go into shock. The heart rate will rise and the pulse will get weaker, leading to death.")
                    st.markdown("---")
                    st.subheader("Why Do Dogs Bloat?")
                    st.write("This question has perplexed veterinarians since they first identified the disease. We know air accumulates in the stomach (dilatation), and the stomach twists (the volvulus part). We don’t know if the air builds up and causes the twist, or if the stomach twists and then the air builds up.")
                    st.markdown("---")
                    st.subheader("How Is Bloat Treated?")
                    st.write("Veterinarians start by treating the shock. Once the dog is stable, he’s taken into surgery. We do two procedures. One is to deflate the stomach and turn it back to its correct position. If the stomach wall is damaged, that piece is removed. Second, because up to 90 percent of affected dogs will have this condition again, we tack the stomach to the abdominal wall (a procedure called a gastropexy) to prevent it from twisting.")
                    st.markdown("---")
                    st.subheader("How Can Bloat Be Prevented?")
                    st.write("If a dog has relatives (parents, siblings, or offspring) who have suffered from bloat, there is a higher chance he will develop bloat. These dogs should not be used for breeding.")
                    st.write("Risk of bloat is correlated to chest conformation. Dogs with a deep, narrow chest — very tall, rather than wide — suffer the most often from bloat. Great Danes, who have a high height-to-width ratio, are five-to-eight times more likely to bloat than dogs with a low height-to-width ratio.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/bloat-in-dogs/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cardiomyopathy")
                    st.image("Canine dilated cardiomyopathy (DCM) is a primary disease of cardiac muscle that results in a decreased ability of the heart to generate pressure to pump blood through the vascular system.")
                with col2:
                    Cardiomyopathy = ''''''
                    st.markdown(Cardiomyopathy)
                with st.expander("See More Details"):
                    st.subheader("DIAGNOSIS ")
                    st.write("DCM is diagnosed by echocardiography, which demonstrates the chamber dilation and indices of decreased pump function characteristic of the disease. Thoracic radiography is useful to evaluate pulmonary (lung) tissue and vessels, and may show evidence of fluid accumulation in the lungs (pulmonary edema) or around the lungs (pleural effusion). Electrocardiography may be used to characterize heart rhythm and to rule out arrhythmias; and in some cases, a 24 hour electrocardiogram (Holter monitor) may be recommended to more accurately characterize cardiac rhythm.")
                    st.markdown("---")
                    st.subheader("TREATMENT")
                    st.write("Treatment of DCM is directed at improving systolic (pump) function of the heart, dilating the peripheral blood vessels to decrease ventricular workload, eliminating pulmonary congestion if present, and controlling heart rate and cardiac arrhythmias if present. These treatment goals are addressed by the administration of cardiac medications, which may be delivered by injection in emergent situations, or orally in patients that are more stable.")
                    st.markdown("---")
                    st.subheader("PROGNOSIS")
                    st.write("Canine DCM can be a devastating disease, and the prognosis for dogs with DCM is variable depending upon breed and status at presentation. The prognosis for Doberman Pinschers with DCM, for example, is less favorable than in other breeds, while DCM in Cocker Spaniels may be relatively slowly progressive. Patients that present in congestive heart failure generally have a worse prognosis than those that are not in congestive heart failure at presentation. Irrespective of this, medical therapy may provide significant improvement in lifespan and quality of life in affected dogs.")
                    st.link_button("Source","https://www.vet.cornell.edu/hospitals/companion-animal-hospital/cardiology/canine-dilated-cardiomyopathy-dcm#:~:text=Canine%20dilated%20cardiomyopathy%20(DCM)%20is,blood%20through%20the%20vascular%20system.")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hygroma")
                    st.image("https://img-va.myshopline.com/image/store/1668653400963/ProP-Bandages-Fig-1.png?w=728&h=437")
                with col2:
                    Hygroma = '''A hygroma is a fluid-filled swelling surrounded by a thick capsule of fibrous tissue that develops under the skin. Hygromas are typically not painful. They can form over any bony prominence on the dog’s body, such as the side of the hock (ankle) joint or over the side of the hip, but they are most commonly found over the elbow. When they first form, hygromas are usually small, soft, and fairly mobile. They may never grow large enough to even notice. If they do grow larger, they can become unsightly or hard to the touch. In the worst case scenario, hygromas can become infected, in which case they are painful and must be dealt with aggressively.'''
                    st.markdown(Hygroma)
                with st.expander("See More Details"):
                    st.subheader("Is a hygroma a tumor? What caused it?")
                    st.write("Hygromas are not tumors. They occur in response to repeated trauma to the tissue over a bony prominence. That is one reason why the elbow is the most common site for hygromas to develop. Especially for large and giant breed dogs, the repeated trauma of lying down on hard surfaces — hardwood, tile, or concrete floors — can produce an inflammatory response in the tissue under the skin over the elbow. The body tries to protect the inflamed area by creating the equivalent of a 'pillow' If the trauma continues, the hygroma will grow larger.")
                    st.write("Hygromas tend to be more common in dogs that are sedentary and spend more of their time lying down, thus increasing the time that pressure is applied over the hygroma site. That said, any large or giant breed dog of any age is potentially at risk of developing a hygroma if they spend time resting and sleeping on hard surfaces.")
                    st.markdown("---")
                    st.subheader("How are hygromas treated?")
                    st.write("The first step in treating a hygroma is to prevent further trauma by providing bedding with adequate padding. Egg-shell foam or memory foam beds may provide the best padding. In areas where the dog enjoys relaxing, the floor can be covered with interlocking foam tiles like the ones found in fitness clubs and day cares. Padded surfaces alone may be all that is required for stabilizing the hygroma.")
                    st.write("There are now commercially available elbow pads designed specifically to protect hygromas (dogleggs™), in order to prevent them from growing or becoming infected and painful. These elbow pads are sized to the dog and then adjustable for a fine-tuned fit. Most dogs will easily tolerate protective elbow pads.")
                    st.write("Should the hygroma grow to an unwieldy size or become infected, it will need to be treated with appropriate antibiotic therapy and may need to be removed surgically. Surgery does not address the underlying cause of the hygroma, so protecting the involved area post-op will be critical for good healing. Your veterinarian will help you to decide how best to proceed with your dog’s hygroma.")
                    st.markdown("---")
                    st.subheader("Can hygromas be prevented?")
                    st.write("There are several things to consider for preventing hygromas. The most important is to ensure that a large or giant breed dog is not allowed to become overweight or obese. Extra weight greatly increases the risk for trauma to the tissue over bony prominences. In addition, it is important to provide bedding with adequate padding. Finally, it is worth covering hard floor surfaces in areas where the dog likes to relax.")
                    st.write("Once a dog has developed a hygroma, attention to a few details can prevent this condition from becoming a problem.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hygroma-in-dogs")
        elif breed_label == "Italian Greyhound":
            tab1, tab2, tab3= st.tabs(["Anesthetic idiosyncracy", "Epilepsy", "Hemangiosarcoma"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Anesthetic idiosyncracy")
                    st.image("https://www.mdpi.com/animals/animals-14-00822/article_deploy/html/images/animals-14-00822-g001-550.jpg")
                with col2:
                    Anesthetic_idiosyncracy = '''A condition where an individual has an abnormal response to commonly used anesthetics sometimes leading to death. Idiosyncratic means there is no good explanation or way to predict this.'''
                    st.markdown(Anesthetic_idiosyncracy)
                with st.expander("See More Details"):
                    st.subheader("Symptoms")
                    st.write(" An abnormal, unreliable response to commonly used anaesthetics. In severe cases it can lead to cardiac and/or respiratory arrest during the surgical procedure with the danger of a fatal outcome. Unfortunately, this reaction is completely unpredictable and there is no certain way to predict or determine this kind of response.")
                    st.markdown("---")
                    st.subheader("Disease Cause")
                    st.write("It is believed to be caused by the incapability of the liver to properly metabolise anaesthetic agents.")
                    st.markdown("---")
                    st.link_button("Source","https://ngdc.cncb.ac.cn/idog/disease/getDiseaseDetailById.action?diseaseId=14")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Epilepsy")
                    st.image("https://canna-pet.com/wp-content/uploads/2017/03/CP_EpilepsyDogs_1.jpg")
                with col2:
                    Epilepsy = '''A brain disorder characterized by recurrent seizures without a known cause or abnormal brain lesion (brain injury or disease). In other words, the brain appears to be normal but functions abnormally. A seizure is a sudden surge in the electrical activity of the brain causing signs such as twitching, shaking, tremors, convulsions, and/or spasms.'''
                    st.markdown(Epilepsy)
                with st.expander("See More Details"):
                    st.subheader("What Are the Symptoms of Seizures?")
                    st.write("Symptoms can include collapsing, jerking, stiffening, muscle twitching, loss of consciousness, drooling, chomping, tongue chewing, or foaming at the mouth. Dogs can fall to the side and make paddling motions with their legs. They sometimes poop or pee during the seizure. They are also not aware of their surroundings. Some dogs may look dazed, seem unsteady or confused, or stare off into space before a seizure. Afterward, your dog may be disoriented, wobbly, or temporarily blind. They may walk in circles and bump into things. They might have a lot of drool on their chin. They may try to hide.")
                    st.markdown("---")
                    st.subheader("How is epilepsy diagnosed?")
                    st.write("Epilepsy is a diagnosis of exclusion; the diagnosis of epilepsy is made only after all other causes of seizures have been ruled out. A thorough medical history and physical examination are performed, followed by diagnostic testing such as blood and urine tests and radiographs (X-rays). Additional tests such as bile acids, cerebrospinal fluid (CSF) testing, computed tomography (CT) or magnetic resonance imaging (MRI) may be recommended, depending on the initial test results. In many cases a cause is not found; these are termed idiopathic. Many epilepsy cases are grouped under this classification as the more advanced testing is often not carried out due to cost or availability. A dog’s age when seizures first start is also a prevalent factor in coming to a diagnosis.")
                    st.markdown("---")
                    st.subheader("What is the treatment of epilepsy?")
                    st.write("Anticonvulsants (anti-seizure medications) are the treatment of choice for epilepsy. There are several commonly used anticonvulsants, and once treatment is started, it will likely be continued for life. Stopping these medications suddenly can cause seizures.")
                    st.write("The risk and severity of future seizures may be worsened by stopping and re- starting anticonvulsant drugs. Therefore, anticonvulsant treatment is often only prescribed if one of the following criteria is met:")
                    st.write("**More than one seizure a month:** You will need to record the date, time, length, and severity of all episodes in order to determine medication necessity and response to treatment.")
                    st.write("**Clusters of seizures:** If your pet has groups or 'clusters' of seizures, (one seizure following another within a very short period of time), the condition may progress to status epilepticus, a life- threatening condition characterized by a constant, unending seizure that may last for hours. Status epilepticus is a medical emergency.")
                    st.write("**Grand mal or severe seizures:** Prolonged or extremely violent seizure episodes. These may worsen over time without treatment.")
                    st.markdown("---")
                    st.subheader("What is the prognosis for a pet with epilepsy?")
                    st.write("Most dogs do well on anti-seizure medication and are able to resume a normal lifestyle. Some patients continue to experience periodic break-through seizures. Many dogs require occasional medication adjustments, and some require the addition of other medications over time.")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/epilepsy-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hemangiosarcoma ")
                    st.image("https://www.mdpi.com/vetsci/vetsci-10-00387/article_deploy/html/images/vetsci-10-00387-g001.png")
                with col2:
                    Hemangiosarcoma = '''Hemangiosarcoma (HSA) is a highly invasive canine cancer. This cancer causes blood vessels to branch, fragment, become leaky, and ultimately rupture. It’s the cause of about two-thirds of heart and splenic tumors, with metastasis (secondary malignant growths) affecting the liver, lungs, lymph nodes, and bones. '''
                    st.markdown(Hemangiosarcoma )
                with st.expander("See More Details"):
                    st.subheader("Diagnosis")
                    st.write("There are no easy lab tests that can diagnose hemangiosarcoma. X-rays and ultrasounds will show the size and location of a mass but won’t definitively tell you if your dog has cancer. The only way to truly diagnose HSA is to surgically remove the affected tissue and send it to a pathologist. But doing that must be done by a veterinary surgeon, as it is complicated and possibly dangerous.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("When a dog is diagnosed with hemangiosarcoma, and it’s too late to treat it, studies show the dog is most likely to live only a few more weeks. “Survival times usually do not exceed one year, even with surgical and chemotherapeutic treatments,” says Sams. I lost Fin within three weeks.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hemangiosarcoma-in-dogs/#:~:text=Hemangiosarcoma%20(HSA)%20is%20a%20highly,%2C%20lymph%20nodes%2C%20and%20bones.")
        
        elif breed_label == "Whippet":
            tab1, tab2, tab3= st.tabs(["Hemangiosarcoma", "Lens luxation", "Alopecia"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hemangiosarcoma ")
                    st.image("https://www.mdpi.com/vetsci/vetsci-10-00387/article_deploy/html/images/vetsci-10-00387-g001.png")
                with col2:
                    Hemangiosarcoma = '''Hemangiosarcoma (HSA) is a highly invasive canine cancer. This cancer causes blood vessels to branch, fragment, become leaky, and ultimately rupture. It’s the cause of about two-thirds of heart and splenic tumors, with metastasis (secondary malignant growths) affecting the liver, lungs, lymph nodes, and bones. '''
                    st.markdown(Hemangiosarcoma )
                with st.expander("See More Details"):
                    st.subheader("Diagnosis")
                    st.write("There are no easy lab tests that can diagnose hemangiosarcoma. X-rays and ultrasounds will show the size and location of a mass but won’t definitively tell you if your dog has cancer. The only way to truly diagnose HSA is to surgically remove the affected tissue and send it to a pathologist. But doing that must be done by a veterinary surgeon, as it is complicated and possibly dangerous.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("When a dog is diagnosed with hemangiosarcoma, and it’s too late to treat it, studies show the dog is most likely to live only a few more weeks. “Survival times usually do not exceed one year, even with surgical and chemotherapeutic treatments,” says Sams. I lost Fin within three weeks.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hemangiosarcoma-in-dogs/#:~:text=Hemangiosarcoma%20(HSA)%20is%20a%20highly,%2C%20lymph%20nodes%2C%20and%20bones.")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:  
                    st.header("Lens luxation")
                    st.image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBUVFBgVFRUYGRgaHCAdGxsaGhwbIB4bGx0bGhsbIxsbIC0kIB0pHhgaJTclKS4wNDQ0GiM5PzkyPi0yNDABCwsLEA8QHRISHTIpJCkyMjIyNTIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMv/AABEIALwBDQMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAAAAAEBQIDBgEAB//EADoQAAIBAwMCBAQFAgUEAwEAAAECEQADIQQSMUFRBSJhcROBkaEyQrHB8AZSFGJy0eEjM4LxQ6KyFf/EABkBAAMBAQEAAAAAAAAAAAAAAAECAwQABf/EACcRAAICAgIBBAICAwAAAAAAAAABAhEDIRIxQQQiUWETMnHxQoGR/9oADAMBAAIRAxEAPwDR6jxHdB/6hGcgW4wM/jUEj2U4qvTeIFl8gvOc8MVH2RWHHQfWj7GpRt3wwh67l2n5ET96ily8wyEGeRvAA9ASBNZy5Bbl1gpFvavX4m4t7ZJPPBiK5dVyvlknuLduQf8AzCj+GpDTzO9p7j4gODOMZ+9DlBbP/StCSILYGB0OCx9DXHErthCGNwh2iWMAcdSFB/2qlNUdpFuySk8pLT67lOD6gV7/ABN8kKwtAESEY5MDlWY5AjjEZqWl1GA29DHO5oA75U7Y91685rjgHxHw1Xts1u24cnCb1Dk5kH4gJHE9T6daCXwsJbAuo6zBA3C4AuPxYX1n9abahLR3QqB2GQyWn3CeRld2Y/N2xNV3/FGKybZJAxhkIHHLE5+c8UTtiqxpbCOk3EAYErschjtjjLFk6kSYPQVJdQrf9va6sC3xLcFlIzvKyJUf3Kw9QDiu3/FV27HndyouKXE9ILZ3AgZ830pVfs2rh3ruVpJLqNh38hyggZyJSD9YoVYVYRrfEArgFirZi4qkg583lkqwPVWgj3o3Uajy7RsKGHbbIOQMicGYODM9GxhF8JkLEkNkGe5AwwIAhvWPej9MPNuE54niPbjiuHottk/mTc3AcxuK/wBjSIcdVY5EZmh1sfDYOkK+7cGyMidw2jIBEY9DTN2A5IyOOJ680LfZOeT1/b7frXdBUbI6fTKQMS8ny5IKmDkDsROO80XfViM+kxmSsDAPBiDPynpQmi16bpII78e3vxTI6uyY7Gfr05Pt9aKYHEnpnPw2Nx2LEDaBGQOCZwBweJzSq/eYLCSvcc9eD1jj796c3tfpcQG8v4trDn+7POKTpftNcw25QTzgxM/M0WxVH6BrWrhdhURPqJgnHeIJHsfSmqa6eS0SJAaJnmCQewPympWtXpfNuTc0eUiBB9RRtldC6/idD3ORPc0P9hr6YrfUlSGV2lVJBuROIO3H4hgwf81E6bxW2hld4YrtnlYHmWQTzgDsJPoaaXP6a3putXFcxMBs/SlL6O5Zb8IDd9vQiCIIg4NFipJ9M0Wm1/xEJ/CwAIiOGBaQeCI+kHmJoiz4kjkI/lMYZTGcfIGf1+VZq1q2LFGCgEiCPKByAJHAjHtPeas0GoUxbuJuMbcGdw4gAcRz75o9iuND29qLlgCU325klBG2TkgdDmcY5HaiHIYblYSchgY3dtw6MDifQg0t0V65bYGdyMYgnO2SDj0gfepiZLIvkP4lxmO09cAfTpxzQLCLXiPwiq3V2o5hT+VWnjd0U4I/t9gCbfENKysXSYIk9dlwDyXI6qRKtHp60Bp9Ud5s3F3WHB2sRAU9UI5AI47fozsuLQCbiUXAJPmQGAAe68ZOM+lCjmCWtexT4ggoPxwQGtsOW/zJme8HtgNtPfVpttBlZB5BXr9z8prP67TtYvfHtZR/+5bAwQJ86r3GZT3juLvjrZuJH/buEBWGQjkeSD/aRK+wSk2htMY+KaVXCyDKxDddvoR+YdPYd6Ct64pcVLhG8rg9HUdvXEgdAx/tputwCVYAgZg9F4+YGRSvxXw1bh+HgBx5D2uLkR2mI9QzU3YvRnP6w0r23XV2xKAAXAOQpYSfbg+6etP/AAbWE2xOYxj0/n79aq8LTeGt3AAtxSGQ/wB2UuLnoSNw/wBQpV4ZZfTNctkFlBAUnssr06wB9qV32hl8MNNvYu5tzkcRAj/yAEZ7A1Re8QllIhB+E8MQB1hsE+gWfWlLf1HvYIpK569jwCSGBPoCODXTrPMp/wAVs6bWUrmcbpWR0GR1rqCG+IpeIXZcLBjg7B8Nu+QDB7qQDVYS6CXVFLbTP4eByvAbr2iOe9RD31fyMroMspl4/wArKMlSOD0n5VRqNOrSyIFjJDs8QeRvBx6TIPcYrjj1rxTUIWgMVXm3cDNjvbcEbs9BJqep8StsR8VHtsQM3F3LHb4oQMvTDRHU0KLi7Ph3LjxMxu6SAp+GwBicTJE95x66MsHCOCvO3YzLj8RT8wEwYgzIPEmji69d+ENrBthPDAXLeMTI8wPqcdfWqDqzub8YSPLBW4J/L18oxyCf2oO1aMBVuMyz5fMQyDJ2n6+oPpRYtQZbMDoI/SgUUQdNAWct5drcgCBjjcvfHPrR9/SqoHBI+dA/4qeD7jM+/tVGo8UHGeKN+BuPktuusGSOc9KrHiNtOxHvz/P2pFcus5MZ9hNDrpQD5mUfOfsKNAb+Bxe8YSCIbNLdR4uT+EntmoFB/nb2WP1oRw04tn5/8VySOcgv/GNEmO5qJ17Dr9zQDrdIxb/Whi9z+37UygI8iGRuuczUU1LDj9aWHVXB+X7VH/GH+CjxBzQ4u61v2q+x4m8ASYFIf8Z7Vfb1Y6/70rgNGezW6T+oHVoVyg75kRMcc/8ANbHw/wDqv4q7L6C4By4wyjiZ9yK+U6e4rMIYCcZxHqfSmIRkBPSY3KwYScgErj1j/alprof2y7Pp93RWbqFrVwTxtcgNziO/agX0txByQwxDT7HnpWL0fiT28YPWf29q2fh39R/EUJcVXQdx5h2huflQUl5OeNpa2MW16mAElwu3OfNAEyT1A6z+9F2gzTd27NhIIkCGG3cAPUmJ4waB1entsDctkFf7SfMk9x29aHTVXPKrMT0Bk5noZ5iBj/KKopfJBwvoYXZLcbG5ZYxkiDxyMwfUiidLqJ/6dzLBSAwyGEwVI9oI9yKDFoOwNyQw2CCMFZ2uwI9l9oorSqpRvK4aQykr1kZntIU/OjVk+iy65C7lAZQfOgyZ/uWeeOOo+4Gp06kNZDeS4m60VPBWHAU9eOP8o/uFMNHDMzKT5gTjgNgY9jNKNS7AoAVHmL2wDhXH4k9jDgf6hS0MmPjqSbYuCNywW9oAuD24b5V0XgbIckAAAnqAUgq/ygSO3tSTQeIg3riplbifEtg94K3E95BxTjw63stujAlASAeTsYAgn1EkH2pBmc1dr4kXrWSQSpGcxiffaB8hWe/qG/qXdLukyHQB1idrr9YndH/jTfQWTpt1tW8k+T/Jukhf9M4HowqFzTXLV25dsJvW9tLJgbHUGTno24H3Bo2DoDs6K0FZQLQg5S2jAg9QQCXY496DW7p5KrbXaq9VaQcjlsr2460btNwBXRkYCME7YIiHIAHQjEx+mU8T1FwXAL42p+EeWbizwQ8sJgY59BR2FUOGtWiRcnZJAaYBnkHKwQQDz1nrVGuUWmjc5DGbZdwWUwSSjTuIwBtaY5xSnUX2tgXLTNd2E7t2Cw8rMpBzxLiO5PYUUuqJtkON6Ya3BhrZIBE9eP0HSJPR1X0D6+0txYa3tdc708y3AeSQRg4ysKQZJB4qNtzCjqBEiRI5yO9TtjdBAicGMA+sDimPlXLc+gqbZeEKKLRAEnpVOp1oYRMVLUOSf+nmenf6UOmjLMZye3Qe5oK2UaSAG3ng46ngfX/aiNN4czHyqW9SCB9OT86caDweXBbP6D2rYaTRKowIiqKJKc0jH2/6ZZh5ifYYH0FF6X+mranK5rYuIECgWfpTUS5titPCraDCChNR4dbHmIFGeJ6opgR8z9eKzmp173JA47+noKrGHyTc34K9fctkFQDnoCI+fWlNxViNoA6CiHEUJzNXUUZ5TfgHuWEPShLuiQ9KPIzUikim0LsSP4Wp9KEueFsODTy4KgVpXCLCpMzj2bi8ianp/EHQ4Zl+Zp/eQFR8wf1/f7UDe0St0qTx30Xc3F/8LtDr0aA52+o4+dafTXwFDKcCQD0n5+4+tYC/o2TIojw3xVrbZ44IPBHz4NQniNWP1Phn0Xw7xo23EyJ6zz7jtWuBt3EVraneJ3LzjqR3618vuXluqbluAo/EkyVnHWCVkjv84JrQf0z4sbbr5iTM5k9fX71JNxdMrKKkria/T/iIkjyn8uOsDPr96Lu/EB88gQIzyJM47R19PWi9KEuruEBuI6EHsOnPSvJeiEuQVIEGBjMxPy+9XSMcuxdpLTpc+Ey/i3NPMESoHqMzj1obUIrBkOCHeDzmZ+oKz86Zai1cW4GgyQ3lnBkeb6MsifWl9y2AS65DHnnnbJM8AmaAooDkMGRcISw28B9ux1+bMG+RrUeH69bibiYAJEHoVO0qf51rNPbcMXRtokHHEyMz1wpz3qjTakq/wy0owlgB5Q7tIgzOYC/+8rJVsZb0bPXWhcUR1GM9DHUfIz0j0qWluuBKKWBgEYBDLzM9wR8waHR0hGU/5SO+DI9OJ9CPU1Es+4xtnE7szyAw94z6g0v2d9GbfUsp2qCCMiCqz3MnBxMml+o1tk2wHV2RWI3sxLICxxMbjbM4M9xgkgt9Xa2grbDxI3Eo9wsTGSXABYgHpisxrle3d3LMkeVW2gwcssEg7TJlQI6URkrOP4UFJdcI8Hnh0ki4pUZ5ggmYJHMUbprDCSGhiNrHkMIjB/tIjFS015CCuULAkiWKFuBH9pjEZHqOKO0lrGRgf7frSSkXhCuyqzodqk4/npQrI5cpIjme3zpk6bzhiAP17VDTaU3X2geQGSf7j29qEU5OijairZHQaJmO1BAOC3U+3pWkseCJbXFF6DSbflRqcx0rUoqKpGOeRydi5NOEG6Mmq7mqKnnFNbqiM0hvKZPaptNHRkn2E39UAJmsxqfGiWIRQVGN3QGrvFb/AJJEnoIMdaS6LTm6QIO0Ge0/8Y+9UUaXKQLXSLkV7jfi8p5g8/PtR76RVwomjbFlUUYz6UZd08rMQajPM3I5R0ZS/pKXX9EVBMY/atcdIAGnJx9qp1NjyndER0708MrIzijHtb4xFcfBAij9W9sHHT9aA+JBJI9q0Rk2JNLimv4f8kGQEn7VF9OMhTwJkmO2M9avsxtNdS3MU5Owf/8An3Ph7tsg8EdSOgHP5qFuWCphgQexEH6UxXBxXn1LTDAOMc+nqO4x86C7aLyp41K15VeRM9uaA1OiB961drR2rhgMLTcQxJUkmME/hHeSaE8U8HuWX2uAeCGXKkHiD+3NFxsmmZawblo7gDtBiegJ6H3E46wa1Oh1SOA6YE5SZj19R+n3ofRXDacttV1YbblthKuh5Uj6EEZBAIyKs8T8I/w4TWaUl9OzcNBa2/Pw3HB9G4YVmyY7NWHM4v6PoH9PeJkkIT7VrNOEZWDQFIyOs8Aj5mvkOm8UAC3EgBuV/tIPHMxwQev1rZ+Ga03UEn3/APVSjJr2s0ZYKXuiP9czsitwFElV5ACliZg+uOu7pzQqCSzgnsRIPlkCSCY/Ef8A7V3R6u1ZQhgSxBjmDOIxgdM0LZ1AHkUhd6qoB4CyJyODg89u1PZm4ss1Fq0S3l856ggYgDicQR9qSam3tdFUblUiHJIU8dB+bg8dvStVds2woOAdsMdpA27iw+sjPvWd8VZC2du6AFUTkzCiDgHaQZMTzzXNaOj2X+HXS9i4rHYQ+6R+Vj5jnkKWDAjsxrur8bVAnxGRTtiHBK+XEqQJ7TPQL1mkl3WMVdFBQ7tzEz5jJOyB6R96r1lpWW3uUYBAmTjB6R360gzG/i6HMnywQB+JpPUEzjHWs3Y06s/4dsD055mMdad+Ip+VQR1Inv7x+lCIhWCFyesfL9qSTNGOOi21pVmYzR1sFVr1iD/l9SDH2rxR1UsQcZx3pO+iv0zr25i2kgtz6DqfnWh8P0vw0wv/AKpJ4LZJYbj1yeT7VrLggADNasceKIZpbo4nbqauQgYqu0IOT9KjfuQJ9aezN9ENU44pDr70g9uKZXHmelZ7xbVBfLj09z1oxjye+gN10LdQ5eLaD/VI7/rTPTWQvlwAOvFR8O0hRdxwTkk9PaibVmfN0M/brUcs+T10hoqkStoD6gcTVum1KuzKZlenocA/P9jVGlTLSeMgn51VctkQy4cd+D3U+hj5EA1n4lItdMaXETZnBGJrG+K+Il/IshR96I8U8Ta4NqyB17z2+tKNlbMOPyzFmk4tryDlK7sotLM9Ks2Aetavx2Tx5XC6XarYALE0SoAWOtWkV0W6ssV9kuRQErhsjtRXw658OjwDzAzZHajvD/EWtjY3ntHm25JXHECfX2qv4dVNbzSuIykW67wi2bbXbTjaGgoZ3Cfwx37EenWhPDNW2ndg6b7TjZdttwy9fZgcg9DRenuFSY4OCOhB5B9Khrre6X6T7nj7jHP1qTjZWMhD474cNJcVrT79PeG623pOUYdHUmCPn1rQf0vrtpI3TiflgEZ+tJPEgxttbB8hYNtx+JZAYTwYJ4pJoNa9tgs8HFZcmOto14sv+LPqutuEw4HHQ8VQl0M4MQJED2ETj1qvwvxFbtoHMjkGuLzjgVBvZpS0bE6tfhs25Q5OegIAVQAD18x+lKdPaUtcuuoxvmfeN2cQMx7UtcmCZMkQFBxMYM+/T0rQR8O0bZdQxkZG7ORk+w6f3CqJ2Z5R4mS8S1YBnaYJLMRgwCQsHOYMduPktbWEAE7m3Et5TAGdsf8A1j5Vf49cG18tDCRO2SFhYmcD8WP8tLbOtUDarYXA4mJMSQRJ69eaDRyNhqb1tZGzBH4d0jEdo/k0Xora3ApiMcZ55xP8xQl57bAAESTBEYgAGQwEcyIj50y8PfAjiot7NSXtD9PpFlZGJEjjFc1OlNxiANoAJIHaYA+9FlsY59cVZpW/F6wD9Zqsasm2+wfw/wAPCDimycYGa8LcCp6fiqkJO3bB3tnk80HdJEyeftTLUHil96ikLz1sW6u5tU1nEtm7dkwQo3fPpz6U819v4h2cL+YjEfWunQCBsETzTZHxjS8k4u3YHbYzAM4k+3airaGFWOn68iu3EVWi6sLwLg/CROFefwnpMwe4mKY6bQhWmTn0rC7ZocaVsEt6TDAiB3pJ41cNvyd+vpWi8b1y27cD8XT/AHrB6i81xizEkmtGOHyZckwd1M4+dXJYjLfSrUQLk81VccmtmPGSyZHJK/Cr+zpecDioqKki1eida1RRBsqW3RC2PSiNPZpnptHuIgU7lQvYmOnqtrHOK1eu8Da2ATBnt+lKn0lLGakrRzTi6Yo+FFUtb7im9zTx0od7FBjJiw26mUnHcUS9uKiF4qbKJifV2Kzfiuj6itvqUmaR6yxPSklG0UjKgL+lfEtj7SecR61rrs4M1841CG3ckVtvBNct5QJ8wrz8kKZ6OLJaHEiV9P5xT7UXbUEs7Mc7CVMGV2lhJyTyPUUhR9rcZEc068U0zG2twuDK+UgKNoBHl2HiAScdx8xFnZF0ZTxtCzbWUsS34XBnzfmxEEyDE8xWZ0emuFrnwiFG4yGZljJhYHUcGad+KamAbbKFOfPBkgjCnukhT1rOWLiqWDqekbY+8/LiiictG4VjPsc8zJiAQQK0Ggc4wP561mtJe2PHmbd+fgfQ1odC5OB/xWetm29Dwt5e389aJ8NIIBXgxQazt4mi/CXBGBESCOxB/wCaeD2SmtMclcVE4qa8VU3WroyMovNiaA1L4mi73FKtdc8pHvV4IlMSa3WGdoMBjn5U00eoLAn8gx8+9IlE3GJMgD+YFEjU2yAqhl7561m9VJ3SLY4KkH29f59pAIzI7/XFdOoa0s2xKR+An8P+gngf5TjsRwRCyiM55+dDeI63yQPaoRtsdz468fAs1/iHxTPqf59KoWEEn8X6VAQPN9B60NcuSZNehihrZiyyi5NpUvBazkmurQ6mavtrWuJmZegoyzbqnTpTCytPYrCNPbrReBsquCR/O9I7XrTPTNFJNclQ0HTs1PiGn3pjkZFZPU2Y6VqPDNTuG0nI4oTxrSY3r86hjlxlxZfJHkuSMpctig7iU0uLQlxPStTMoruJVKpzTFrVUNbg0jHTBHfaVYciD9DSvXoCzRMEkieY6U01QA+lLLxmuoomBaPwO3qC++4tsosruiGaGYJJ4nac9470h01x9NcDcCadAy4QDMlj7AED/wDR+grRabw61qLB072wLpM2367owhPY8D1juawZv2o24bUbA7OqDwwOTTbR68LIuAOvJ3cgjLEHn75rAXTc0rlCDA6dR6VDXeOkqQFieeazKLs1SlFo0/8AWKW2QmyJ2ZLbt4gxHAAESAfXisNZ1cTIB96kmsJQrQBqqIM+naPeqnft3cRABA6dOaY6HXFXmPKeeKXvbYbSclszu3wP7y3Y9AJ/SZ6fucLMCetZ5G/HTVG60WpUir9FqENwhZieojnrWc8NukqQDRvh19vjEMR+8/wV0ZXQuSFWbBHxVQPpXbOancOK0xMMgDUNGazHi2pP5f5860WrfFIPEbPlwc/z/er4nciM1qxDbaN5nOBzQ9u/yalfO129f9qDt3YJMY/2pckNtlYSuhzbvL5T1mDP2/npS/UPucgHE0Bf1Y6DrU0eFnv+9Tx4yWWWzupuZgcCqC1cc1Hmt0FSMsmXWxR1i3Q2mSmdlIFUsmy+2ooi3VKCiUNADL0o60Tig0om2aY5DLS3ipBHSidXr2cRwOwpdbaasYVNpXbHUnVAl5aqKzRF2h3NO3oAO61TcTrV1yuTilZyFXiaeaOwH+9K7lvysewk/wA+dM9UZJnrSbxu4qISD/PnXN0h47YosXDvZh08o9/5+laDTaxkVSTDDIPY9M/elP8AT9reIOJznvRXjjC2kDJry8k3KTZ62OKjFIHs+Lo+oe9qVF0Fsq2N09fLEEc+8Uu/qbR2i7vZDqhgorjzQwJPyng9QRS65qIhoEyDB4MHiOooo3nuN5iWZz7kk+n7V0RZ0JdkJ9qE3Vo9ToBEMQInHrSS8sGBxTkz6z4hat22YTudjkiXGMgCY94JPHXFCOu4DDSQOeh6wBgVd4m4WSXZxBGwN8NR6ABfMfTdQGiDAedXU8w0A/TFQltGvE6Y70n/AEws8daN1CW1RbqnzLcIJ74GDSYPIicdB+tHW0m2wEiYwI5HHNTVI0zTdM2ui1SsitIg9aJd5ECsh/TGuhnsONrKcA/cVpd/QVojK0YckKlQFrJpVqXO3J9J6061KnnkUovZG0Yzj3quL9iGX9TM6tIgnM0vuHbTzxbTkKDjHb+c0g1WJrVkjbIQl7SgNuxRDChtMvm96IuGliqYs2Vdaki5qs1dYFWRBh+mSjVqi0IFXrTCMuQ1ctUIatTmjQgXbNEoaERqIttRCg/TJRdy3ihLFyiLt/FQk3ZVJUB3aFuNV96gL98DrVAMlE5NVai5FBajxLtS5r9y421ZJ9P5ig2lthjFvSCNTeA6/esx4neDuqsDt3CSB0nzfb9K0a+DMfxNB9BMfWg9T/TImd7E+sf7VmyZotNI14sDTTZK5qrKlxYt7ULShIO8LEBTng8569aUai2WbzsF9zP2/amVrwcL/cR1yR/7ptY8KtAbvhmOpZ1I+gAP3rF2zf0jGW/D2eVtWy7HE/OTCice/wBKKs/05cQgXQSedqZbHoJj51qtR4ntG2z5Y42Db6dMsfel1p7zk2pIByQZEz3YCD69cxVLIteWK20KhpYeUdBJbjMngc9O1UJ4MJJYEScRj9c0y1OgbftMEjI2kkdx7T0/aj9PobbCbjXR2hQZ7k+YfvXHUqNFf0JYEhgARhlLufkYED0JrGam+bYCq0tksxyST1M/tPTNavW+Haad1uFJMkRMk5zuzPz70j8RsJ5k8iMcjy/D6zEu6+mRUU0VVrYB/j5EbSDyW7/Pt6U30Go3AGcY49PWs/dtkSMtBzt83HMEE+vWp6LVspj8OKDiaIZV0zT6nS5+KjCUziQT9P0rReFeJi4onB4Mjr86y2h14JgyR6Dr1rtwMhNy2GHUwce9cnQJxtG4a6IoFwN0gYpVoPGRcADEK36/WmFtpyKvCe0Y8mOkxf4za8p+tY/Vda2/igBX3GaxuszXoy2kzDDpoq03epusmoaZSKsDZqaOkUsMxROnXNDxmjNLVURkHIKtFVrVgpkTZNauSqBVoNMKE2zV6GKCW8B1qDasdKVsZJsZi/FVXNYBSS/4hHWgke5dPkE1OU0i0MUpaNs3iGktoGdviORO0SFB7HvWQ1Gqe652LMngDAn9BRWn8IAzcO89gcD96JW+Au1BtjoogVllnUejbD0zfZTp/A+txpxMCYHz70ys2VUQoA/nOKrsah4g4AjnnNXpc2uCGG6AQZgAj1qLm5F1jUdF1pDO/aIQiQwJHzUdKBuON8FomYxIMdok5mOKOtau4A3mCqxjkkE92GSZJA+nNL9Rad2YAgBQY8vlnHb8pkfQ0p3TKH1gkZjoAVnPSQRHvPHal+vuCN3xCQ3YccyR6HvPWrToXmWkgGYHcAhYjgQevarjpLjKQx2vABnnaxYiZHHJPt7VyRzl8C65o9oEqwZ3Vd5KiIgySPWIPEU00qAl1OTlgSPUeXIOMKPrEzm5NJBs7mGxj8N+On4WB9dkD3qzxDThLgV3CoGAIWT5XaEHuCm45/NHFMI9gOisLdCoQqpPmPOSIwxzu3SMmB+ja1oEiZQ9JZ4mMYG5fr19OvPD9EplCVAH5TOCrOrGMAif1FGF2Zj8IBlETODJEk4xmY/8TRQGy6+ltgGe2rqcbgJI7TGSPvVN7SW1/wBJ4JBYT2kyB7ECrbGqtXCWtMM8j19uQajcWQRI45HJ6QQOfcfQVkjI0NGY8b0cnyMs9iQogzPl2gR6iszd0jKTu2YPTJPtnj1r6EmlUTtSZzkhlx1BMxnpIpB414e73QVUK0Z27gCPaW/TNWTEE+n1LKAVmOvbtnsab6HVwJn3HPvSfUsBO6fiLnGQcgCQUHEf+6HuXWjfGDjBMSOhBzPXtXOHlFYZdUzQ/wCBW6SymGAgCZEd/rUbHiNy0IY7xwDx+uKWaXVkQN0NOYpi3i/5SoI6Y60q0UaUkXa/xP4iQpz1B5pLp1Jmc1I6UOxZWA68d+gqL2rijGe//vmtWPOtKRkyem7cSS3IMGrmtsMgYoK9fPFy0R6iqjr9uAx+eKs8kX+rMzxSXaDp9Iq2zcAPNK18U9QfeoNrAeg+VMslEpYbNGL6968dWorNnW9hUT4gaP5UhPwM03+K7VFtWBy1Zg624cAGiLdlzl22j6/SllnXyVh6ZvwNrniA6Zquw126YRT7nAorw2wlvzEbp7n9hTf/ABO6AsAHAxHvxmoS9Q30aY+lS7FA8Ihv+o2/EkLgf80ZbeBCDaPTgVG+WAypM4x6/uYNX+G+HtcQOY2B4YQcDkE98TgdqhJykzRFQgiy2vGfUniBk/tU9PpH3xBXAkTJgiTjngxHypo+oXf8TYvJgFRjgHy8RyBzxnrR2mtRcO1IYg5ImJyTJE/L0PFMoCvIwO3preIDErILdGwME98jA4+dQ8RYOQy7RGG8o/MYMeg4jpiiLyTbcLyCeD+YcwemMwPvU9NoybUNyTuyc9T8oijRLluwHUW/KARI5x142/cj5VIWQYyAVncepHtEcnk9vodf0fw1G4hmcwNwnzGAi44GB9K94XbYrcBEhDtDFfxNMlvUAjj0rjnsEQBQyncfOz8Yi3gCfcgx6xXFtfELqoJLLDk4IJUifkGER3Jpj8X4du5cbKLgA8nbMk+rOT9PSi/CbZVQX/EV3P33Nk/QTXWcJPGraoli1BLOVk9hkhj6j9ZpalxTeuBt+xyIdkJRhHmUnIBkmCRyCOtN/HtQsq24hmYbQFBPl/CCeACYb2ApJ4frHe43mIKnbsLQXAgSUIiOTxIJoUchr4ctySh2Er3UyR8jMEAebMxV2s0Z3f8AaB9AzQDxiI6AcgcUr1F9lcIQXBYAHayjzcQAeese9CX9Rq1dlXYADjcGOCAREtSu0FbGNzQhiWwjRuV13D/84YT6Y7URpX3ALe2h+j2zj0ORg/Y9xxSL+i/Erl9EW6d3SeDgiDI6+tO7o+zbfcEZkcfaoVTov2rDHtuQQ4DqOqSG+YmQY+tBWbV0qTZuBo/+O5O4f+WGB9wfnRGlJ+JbWTDWg4M5UnkKedvoZplc0q3lKvIiCGUlWnvIp46JyZlNd4cpktYdHMjdO5cgmZwRkzJWslqdKFdlUb1n+4EwJHIPqa3C6y5bum38RmXjzQT9gM0O1z40m4qkq2MRwccU6YKoxZO3crIVI4MZntE4EdfavM7MAd0/l28EACZmI+/StRrvArO13gzM89fN8/vWTtW8K25pkjnpTUMpsa+HgJDsIBExkY45+RNWpfBbB8vMCPWl9pVZralRECec++a9p7hAxj2A7UOJRZA29fJMYgVNNPaaJt88yBS4OdrGczHyqKX23DMeaMdqFDNjVvCtMv4lExgA8VS+gtcqgH8PeoM5k1faHl5Nc2wUkefw60o3ATgdDg5xnrx9anp2tDlFHqAJ9881FGJCyTktP/iBFD6CyHu2w0kFs0yQrYM7qHOwTzE/Y/ep6NCzeZSQCCQuTBMYpl4zoESCsjcuRiDILHp8vanPgGht73bbm2AyjpKKzCe4n9BRrYnPVilLJCtAzIAJxlo49ciPY0fo9Ld2ozQEQ+VSAciTJX82T17joKfiyqBmABZF+IC2ZdyASfYEwBAE1Yqht0gSGiesSB/z70apk3kbFul8Ge7cL3ICqhEerLJAHA/F9abfD+GUGwbQZYtJERtABPJ4MAflM4q2/YGw85eTn/Tj2pZdfdduLAAQNEDtHMzM9abom22Fslt2lWEbvPI3T+aJBwOOvbgDJZLSEWeJLdzIhR2B+mPehkPnjoJPAycc96Ov/wBvGBkc55zROKk0gAAmZYwI6n8RHpzJo5tIBBPAznvtjj2AxVyINzekKP8ATEkfUCp3LYYhzyokdp7x3pWcBG1BGJblVPeOZ75yfX5C29p3G1Ejby0zPTHqec+1HFBBPWgrINwKWY4BMCADzg4mPnXUccM4G0bBACjr6kn8o7CSTQGvulAIIJJG5UEl2OFWeg4z6UQ/m+MDwg4BInHWP2ihdG0aYaiAbm3E/hX/AErwKDOQtvaW420vElpFtV4LHkmSAAMljxmM5phY0iSwC7RPPHyE+nI9vWo6G61yNxOZJjHEYjgDzdOwqfiFz4dkMoG5nAkiSJaMew4oIZiLxXTWzuG52FvzbQYO9j5UnrPUdAvrQXgdy8TdF1N8MI3Dgmd0T0mMjmK0r6dSGERDhB7dTmfN60B4xFsqAJ5EsSTiIzPrS34G6P/Z")
                with col2:
                    Lens_luxation = '''A painful and potentially blinding inherited canine eye condition. Lens luxation occurs when the ligaments supporting the lens weaken, displacing it from its normal position. Signs of lens luxation may include red, teary, hazy, or cloudy, painful eyes. PLL can cause eye inflammation and glaucoma, particularly if the lens shifts forward into the eye. '''
                    st.markdown(Lens_luxation)
                with st.expander("See More Details"):
                    st.subheader("Causes")
                    st.write("The lens is a structure in the eye located behind the iris (the colored portion of the eye) responsible for focusing light onto the retina for visualization. It is suspended in the eye by multiple ligaments called zonules. PLL is caused by an inherited weakness and breakdown of the zonules, displacing the lens from its normal position in the eye. The direction that the lens luxates can be either forward (anterior) or backward (posterior). Anterior lens luxation is the most damaging and considered an emergency as it can rapidly increase pressure inside the eye, known as glaucoma, causing pain and potentially blindness. Posterior lens luxation leads to milder inflammation, and glaucoma is less likely to develop.")
                    st.write("PLL most commonly develops in dogs between the ages of three and eight. However, structural changes in the eye may already be evident at 20 months of age, long before lens luxation typically occurs. Both eyes are often affected by PLL, but not necessarily at the same time. This differs from secondary lens luxation, which can more commonly only affect one eye and is usually caused by a coexisting ocular disease such as glaucoma, inflammatory conditions of the eye (uveitis), cataracts, eye trauma and eye tumors.")
                    st.markdown("---")
                    st.subheader("Diagnosis")
                    st.write("Early detection of lens luxation is crucial. Your veterinarian will diagnose primary lens luxation by performing a complete eye exam. They may measure your dog’s eye pressure for secondary conditions like glaucoma. You may be referred to a veterinary ophthalmology specialist where additional testing could include an eye ultrasound to evaluate the internal structures of the eye.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("Treatment options vary by stage of disease and position of the lens. When diagnosed early, the most common treatment for anterior lens luxation is surgery to remove the lens by a veterinarian specializing in ophthalmology. Topical eye medications may be needed long-term, even after surgery.")
                    st.write("If glaucoma develops suddenly, this requires emergency management and may include medication to decrease eye pressure, followed by referral to a veterinary ophthalmologist. If the eye has uncontrolled glaucoma, is permanently blind, or there is pain or inflammation, it may be necessary for the affected eye to be surgically removed (enucleation).")
                    st.write("Treatment for posterior lens luxation may include topical medications to help prevent the lens from shifting forward and causing more severe damage to the eye.")
                    st.markdown("---")
                    st.subheader("Outcome")
                    st.write("Primary lens luxation most commonly progresses to affect both eyes. For this reason, regular and in-depth ocular examinations are recommended in at-risk dogs. Anterior lens luxation left untreated or not addressed immediately often has a poor prognosis for saving the eye.")
                    st.write("Dogs that receive surgery early for anterior lens luxation can often preserve some vision but may have diminished vision that is more blurred up close. However, this doesn’t generally appear to affect everyday life. Surgery is not without risk of complications, and often, patients require lifelong topical eye medications.")
                    st.markdown("---")
                    st.link_button("Source","https://www.vet.cornell.edu/departments/riney-canine-health-center/canine-health-information/primary-lens-luxation#:~:text=Lens%20luxation%20occurs%20when%20the,shifts%20forward%20into%20the%20eye.")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Alopecia ")
                    st.image("https://images.wagwalkingweb.com/media/articles/dog/focal-alopecia/focal-alopecia.jpg")
                with col2:
                    Alopecia = ''''''
                    st.markdown(Alopecia)
                with st.expander("See More Details"):
                    st.subheader("Why Dogs Get Alopecia?")
                    st.write("When a dog scratches constantly without stopping, it causes stress and anxiety in the dog and the owner. To make matters worse, tearing at the skin to try to ease the discomfort can trauma and create wounds on your dog’s skin. To provide some relief and/or alleviate the condition, you’ll need to determine the underlying cause.")
                    st.write("A dog can acquire alopecia from a parasitic infestation of fleas, lice, mosquitoes, or mange mites such as Demodex or Sarcoptes. Spider bites or insect stings can wreak havoc on the skin, too. Plus, an inadequate diet, food allergies, or an outdoor, dirty, hot, or moist environment responsible for a fungal or bacterial infection will cause ringworm or skin allergies. You will notice that a dog will lick and scratch incessantly to relieve the irritation.")
                    st.markdown("---")
                    st.subheader("Diagnosing Alopecia")
                    st.write("In some cases, hair loss can signal a severe underlying condition. A trip to the veterinarian’s office will help pinpoint the problem. The vet can perform a physical examination and examine a dog’s hair follicles for signs of damage. You may also need to do blood tests or biopsies can confirm or eliminate medical causes.")
                    st.write("Diagnostic laboratory tests with smears and a skin culture can reveal any bacterial, fungal, or yeast infections, whereas a skin scraping can rule out parasites.")
                    st.write("Some types of alopecia are preventable, while others are not. If genetics or an auto-immune disorder is the reason for the hair loss, there’s no way to prevent it.")
                    st.write("Ridding a dog of parasites is easier, as many preventive medicines are available. You should also reevaluate your dog’s current diet and switch to a well-balanced one or eliminating common food allergens will improve hair loss caused by an inadequate diet.")
                    st.write("There is also a wide range of prescription medications available to treat alopecia from reoccurring. These include antibiotics, antihistamines, antifungals, and steroids. Your veterinarian will determine the best treatment for your pet.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/alopecia-dogs-dog-losing-hair/")
        
        elif breed_label == "Ibizan Hound":
            tab1, tab2, tab3= st.tabs(["Anesthetic idiosyncracy", "Cataract", "Cryptorchidism"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Anesthetic idiosyncracy")
                    st.image("https://www.mdpi.com/animals/animals-14-00822/article_deploy/html/images/animals-14-00822-g001-550.jpg")
                with col2:
                    Anesthetic_idiosyncracy = '''A condition where an individual has an abnormal response to commonly used anesthetics sometimes leading to death. Idiosyncratic means there is no good explanation or way to predict this.'''
                    st.markdown(Anesthetic_idiosyncracy)
                with st.expander("See More Details"):
                    st.subheader("Symptoms")
                    st.write(" An abnormal, unreliable response to commonly used anaesthetics. In severe cases it can lead to cardiac and/or respiratory arrest during the surgical procedure with the danger of a fatal outcome. Unfortunately, this reaction is completely unpredictable and there is no certain way to predict or determine this kind of response.")
                    st.markdown("---")
                    st.subheader("Disease Cause")
                    st.write("It is believed to be caused by the incapability of the liver to properly metabolise anaesthetic agents.")
                    st.markdown("---")
                    st.link_button("Source","https://ngdc.cncb.ac.cn/idog/disease/getDiseaseDetailById.action?diseaseId=14")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cryptorchidism")
                    st.image("https://iloveveterinary.com/wp-content/uploads/2021/05/Cryptorchid-Chihuahua.jpg.webp", width=200)
                with col2:
                    Cryptorchidism = '''the medical term that refers to the failure of one or both testicles (testes) to descend into the scrotum. The testes develop near the kidneys within the abdomen and normally descend into the scrotum by two months of age. In certain dogs, it may occur later, but rarely after six months of age. Cryptorchidism may be presumed to be present if the testicles cannot be felt in the scrotum after two to four months of age.
                            '''
                    st.markdown(Cryptorchidism)
                with st.expander("See More details"):
                    st.subheader("If the testicles aren't in the scrotum, where are they?")
                    st.write("In most cases of cryptorchidism, the testicle is retained in the abdomen or in the inguinal canal (the passage through the abdominal wall into the genital region through which a testicle normally descends). Sometimes, the testicle will be located in the subcutaneous tissues (just under the skin) in the groin region, between the inguinal canal and the scrotum.")
                    st.markdown("---")
                    st.subheader("How is cryptorchidism diagnosed?")
                    st.write("In cases of abdominal cryptorchidism, the testicle cannot be felt from the outside. An abdominal ultrasound or radiographs (X-rays) may be performed to determine the exact location of the retained testicle, but this is not often done before surgery, as it is not required to proceed with surgery. Typically, only one testicle is retained, and this is called unilateral cryptorchidism. If you have a dog that does not appear to have testicles but is exhibiting male behaviors, a hormonal test called an hCG stimulation test can be performed to see if he is already neutered.")
                    st.markdown("---")
                    st.subheader("What causes cryptorchidism and how common is it?")
                    st.write("Cryptorchidism occurs in all breeds but toy breeds, including toy Poodles, Pomeranians, and Yorkshire Terriers, may be at higher risk. Approximately 75% of cases of cryptorchidism involve only one retained testicle while the remaining 25% involve failure of both testicles to descend into the scrotum. The right testicle is more than twice as likely to be retained as the left testicle. Cryptorchidism affects approximately 1-3% of all dogs. The condition appears to be inherited since it is commonly seen in families of dogs, although the exact cause is not fully understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of cryptorchidism?")
                    st.write("This condition is rarely associated with pain or other signs unless a complication develops. In its early stages, a single retained testicle is significantly smaller than the other, normal testicle. If both testicles are retained, the dog may be infertile. The retained testicles continue to produce testosterone but generally fail to produce sperm.")
                    st.markdown("---")
                    st.subheader("What is the treatment for cryptorchidism?")
                    st.write("Neutering and removal of the retained testicle(s) are recommended. If only one testicle is retained, the dog will have two incisions - one for extraction of each testicle. If both testicles are in the inguinal canal, there will also be two incisions. If both testicles are in the abdomen, a single abdominal incision will allow access to both.")
                    st.markdown("---")
                    st.subheader("What if I don't want to neuter my dog?")
                    st.write("There are several good reasons for neutering a dog with cryptorchidism. The first reason is to remove the genetic defect from the breed line. Cryptorchid dogs should never be bred. Second, dogs with a retained testicle are more likely to develop a testicular tumor (cancer) in the retained testicle. Third, as described above, the testicle can twist, causing pain and requiring emergency surgery to correct. Finally, dogs with a retained testicle typically develop the undesirable characteristics associated with intact males like urine marking and aggression. The risk of developing testicular cancer is estimated to be at least ten times greater in dogs with cryptorchidism than in normal dogs.")
                    st.markdown("---")
                    st.subheader("What is the prognosis for a dog with cryptorchidism?")
                    st.write("The prognosis is excellent for dogs that undergo surgery early before problems develop in the retained testicle. The surgery is relatively routine, and the outcomes are overwhelmingly positive.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/retained-testicle-cryptorchidism-in-dogs")
        elif breed_label == "Norweigian Elkhound":
            tab1, tab2, tab3= st.tabs(["Distichiasis", "Hip dysplasia", "Seborrhea"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Distichiasis")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/2136//2023_2135i_distichia_eye_6021.jpg")
                with col2:
                    Distichiasis = '''A distichia (plural distichiae) is an extra eyelash that grows from the margin of the eyelid through the duct or opening of the meibomian gland or adjacent to it. Meibomian glands produce lubricants for the eye and their openings are located along the inside edge of the eyelids. The condition in which these abnormal eyelashes are found is called distichiasis.'''
                    st.markdown(Distichiasis)
                with st.expander("See More Details"):
                    st.subheader("What causes distichiasis?")
                    st.write("Sometimes eyelashes arise from the meibomian glands. Why the follicles develop in this abnormal location is not known, but the condition is recognized as a hereditary problem in certain breeds of dogs. Distichiasis is a rare disorder in cats.")
                    st.markdown("---")
                    st.subheader("What breeds are more likely to have distichiasis?")
                    st.write("The more commonly affected breeds include the American Cocker Spaniel, Cavalier King Charles Spaniel, Shih Tzu, Lhasa Apso, Dachshund, Shetland Sheepdog, Golden Retriever, Chesapeake Retriever, Bulldog, Boston Terrier, Pug, Boxer Dog, Maltese, and Pekingese.")
                    st.markdown("---")
                    st.subheader("How is distichiasis diagnosed?")
                    st.write("Distichiasis is usually diagnosed by identifying lashes emerging from the meibomian gland openings or by observing lashes that touch the cornea or the conjunctival lining of the affected eye. A thorough eye examination is usually necessary, including fluorescein staining of the cornea and assessment of tear production in the eyes, to assess the extent of any corneal injury and to rule out other causes of the dog's clinical signs. Some dogs will require topical anesthetics or sedatives to relieve the intense discomfort and allow a thorough examination of the tissues surrounding the eye.")
                    st.markdown("---")
                    st.subheader("How is the condition treated?")
                    st.write("Dogs that are not experiencing clinical signs with short, fine distichia may require no treatment at all. Patients with mild clinical signs may be managed conservatively, through the use of ophthalmic lubricants to protect the cornea and coat the lashes with a lubricant film. Removal of distichiae is no longer recommended, as they often grow back thicker or stiffer, but they may be removed for patients unable to undergo anesthesia or while waiting for a more permanent procedure.")
                    st.markdown("---")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hip-dysplasia-in-dogs/")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Seborrhea")
                    st.image("https://cdn.shopify.com/s/files/1/0073/3334/7426/files/image3_79bb87e9-2607-4f97-aa3d-91ac9727144e_600x600.jpg?v=1691678381")
                with col2:
                    Seborrhea = '''In dogs, seborrhea is a skin disease that is characterized by a defect in keratinization or cornification of the outer layer of the skin, hair follicles, or claws. Keratinization is the process in which the protective outer layer of skin is being constantly renewed by new skin cells.'''
                    st.markdown(Seborrhea)
                with st.expander("See More Details"):
                    st.subheader("Signs and Diagnosis")
                    st.write("A diagnosis of primary seborrhea is reserved for dogs in which all possible underlying causes of seborrhea have been excluded. Most dogs with seborrhea have the secondary form of the disease. The most common underlying causes are hormonal disorders and allergies. The goal is to identify and treat these underlying causes. Allergies are more likely to be the underlying cause if the age of onset is less than 5 years. Hormonal disorders are more likely if the seborrhea begins in middle aged or older dogs. A lack of itching helps to exclude allergies, scabies, and other itching diseases. If itching is minimal, your veterinarian will seek to exclude hormonal disorders, other internal diseases, or other primary skin diseases. If itching is significant, allergies, scabies, and fleas will also be considered by your veterinarian.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("Treatment is needed to keep your dog comfortable while the underlying cause is identified and secondary skin diseases are corrected. In addition to treating any secondary infections with antibiotics, medicated shampoos are often used to help control the seborrhea and speed the return of the skin to a normal state. Medicated shampoos can decrease the number of bacteria and yeast on the skin surface, the amount of scale and sebum present, and the level of itching. They may also help normalize skin cell replacement.")
                    st.link_button("Source","https://www.msdvetmanual.com/dog-owners/skin-disorders-of-dogs/seborrhea-in-dogs#:~:text=In%20dogs%2C%20seborrhea%20is%20a,renewed%20by%20new%20skin%20cells.")
        elif breed_label == "Otterhound":
            tab1, tab2, tab3= st.tabs(["Hip dysplasia", "Platelet disorder", "Sebaceous cyst"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hip-dysplasia-in-dogs/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Platelet disorder")
                    st.image("https://images.wagwalkingweb.com/media/articles/dog/clotting-disorders-of-the-platelets/clotting-disorders-of-the-platelets.jpg")
                with col2:
                    Platelet_disorder = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Platelet_disorder)
                with st.expander("See More Details"):
                    st.subheader("What is Platelet Dysfunction?")
                    st.write("When platelet concentrations are below the normal 40,000 per microliter of blood, bruising and bleeding can occur with your dog. Platelet dysfunction is a description of a low blood platelet count for your pet. If the count is too low, your dog may be at risk of spontaneous bleeding episodes. Any canine regardless of breed can suffer from this condition and the severity depends on the how low the numbers are. Usually, the lower the number the more likely the bleeding will occur. ")
                    st.markdown("---")
                    st.subheader("Symptoms of Platelet Dysfunction in Dogs")
                    st.write("Large hemorrhages under the skin in areas such as the abdomen")
                    st.write("Small red dots on pale parts of the skin, gums or even the eyes ")
                    st.write("Pale colour within the mucous membranes")
                    st.write("Prolonged bleed after an injury that is hard to stop")
                    st.write("Blood in the urine")
                    st.write("Blood in the feces")
                    st.write("Dark bruises on the skin from unknown causes ")
                    st.write("Eye hemorrhages in severe circumstances ")
                    st.write("Persistent nose bleeding ")
                    st.write("Weakness")
                    st.write("Lethargy")
                    st.write("Lack of appetite ")
                    st.markdown("---")
                    st.subheader("Causes of Platelet Dysfunction in Dogs")
                    st.write("There are four main body processes that can result in this platelet dysfunction in your dog ")
                    st.write("Number one is a smaller number of platelets being produced by your dog’s bone marrow ")
                    st.write("Secondly, the immune system can be confused and attack the platelets, because in its compromised position it doesn’t recognise the platelets so it tries to remove them ")
                    st.write("Another cause of this condition can be because your dog’s system is using a larger than normal number of platelets in the clotting process which depletes the number available ")
                    st.write("And finally, the bodily system’s overzealous removal of the platelets from the general circulation")
                    st.markdown("---")
                    st.subheader("Diagnosis of Platelet Dysfunction in Dogs")
                    st.write("Diagnosis of this condition requires a thorough physical examination of your dog to check his health and to see if there are any signs of other diseases. Your veterinarian will ask you about your dog’s history, and then run a series of tests to narrow down the cause of this condition. These tests include immune system function tests, bone marrow aspiration, x-rays, hemogram including a platelet count, coagulation profile to test the clotting process, and serum biochemistry tests to check the status of your dog’s general health and whether there are any abnormalities in the organs. Your veterinary specialist may also do tests specific for infectious diseases, or a urinalysis to detect infections or protein loss. While this seems extensive, it is necessary as you want to refine the cause of the problem to enable effective treatment. While this type of disease cannot be cured, it can be managed and your dog can lead an otherwise healthy and happy life.")
                    st.markdown("---")
                    st.subheader("Treatment of Platelet Dysfunction in Dogs")
                    st.write("Platelet dysfunction in your dog cannot be cured and there is no vaccination but it can be managed, although it depends on the severity of the condition. Mild bleeding may be managed by applying pressure to bleeding wounds, and maintaining the pressure for a longer period, allowing the blood clotting to occur. Severe bleeding may require your dog to have a blood transfusion of fresh platelets, (from blood or platelet rich plasma). This is not a cure all procedure, it will assist at the current crisis, but as platelets only last for approximately 8 days, it is not a forever cure. A lot depends on the type of platelet dysfunction your dog has.")
                    st.write("Changing any medications your dog is on may improve the condition if it is an acquired type of platelet dysfunction (some cancer medications can trigger platelet dysfunction). Management of a disease that can sometimes trigger this condition will also help (such as spotted fever or ehrlichiosis). Your veterinarian specialist will be able to advise of the specific treatment once he has isolated the cause and severity of this type of ailment.")
                    st.markdown("---")
                    st.subheader("Recovery of Platelet Dysfunction in Dogs")
                    st.write("Dogs that have a fairly mild type of platelet dysfunction can lead happy normal lives, although care when they are injured is important. Stemming any blood flow may always need your help to apply firm pressure for a length of time, to allow the blood clotting to occur effectively. Keeping your dog healthy with a quality diet and good living conditions certainly helps. Most veterinarians would advise against breeding a dog with this condition as it could be passed onto the young. Taking precautions to ensure breeding cannot occur is the easiest form of prevention. Letting any new veterinarian know about your dog’s condition if an accident occurs will save time and possibly your dog’s life if they are aware that your dog may need a transfusion in the case of surgery.")
                    st.markdown("---")
                    st.link_button("Source","https://wagwalking.com/condition/platelet-dysfunction")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Sebaceous cyst")
                    st.image("https://toegrips.com/wp-content/uploads/cyst-dog-leg.jpg")
                with col2:
                    Sebaceous_cyst = '''Usually harmless, sack-like growths with a lining that is not cancerous. In dogs, most cysts develop from hair follicle structures. These cysts typically appear as single, clearly defined, firm or soft lumps located in the skin or just beneath it. '''
                    st.markdown(Sebaceous_cyst)
                with st.expander("See More Details"):
                    st.subheader("What causes sebaceous cysts in dogs?")
                    st.write("One primary cause is the blockage of sebaceous glands, which can result from a buildup of debris or other obstructions. Additionally, genetics can play a role, with certain breeds being more predisposed to developing these cysts. Finally, trauma or injury to the skin can result in the development of cysts due to the disruption of the normal skin structure and subsequent inflammation or infection.")
                    st.markdown("---")
                    st.subheader("Where are sebaceous cysts usually located?")
                    st.write("Sebaceous cysts can appear in various locations on a dog's body, with the most common areas being the head, trunk, neck, or upper limbs. However, they may also develop on other parts of the body.")
                    st.write("In some cases, multiple follicular cysts have been found in the ear canal and around the rear end. Generally, these cysts occur as single lesions, but they can also appear in multiple clusters or spread out over a larger area. In young dogs, multiple follicular cysts may develop on the top middle part of the head, possibly due to a congenital origin. Additionally, cysts can form on pressure points, such as the elbow, as a result of ongoing trauma or pressure.")
                    st.markdown("---")
                    st.subheader("How are sebaceous cysts diagnosed in dogs?")
                    st.write("Diagnosing sebaceous cysts in dogs typically begins with a thorough physical examination by your veterinarian. They will inspect the cyst and surrounding skin, noting its appearance, size, location, and any signs of inflammation or infection.")
                    st.markdown("---")
                    st.subheader("What are the treatment options for sebaceous cysts in dogs?")
                    st.write("Surgical removal of sebaceous cysts is often a definitive solution, but it may not be practical in cases involving numerous cysts. Additionally, since these cysts are typically benign, simply observing and monitoring them without immediate treatment is also a viable option.")
                    st.write("It's important to avoid manually squeezing the cysts, as rupturing their walls and causing the contents to leak could lead to foreign body reactions or infections. Ensuring proper care and following your veterinarian's guidance are crucial in managing sebaceous cysts in dogs.")
                    st.markdown("---")
                    st.link_button("Source","https://www.kingsdale.com/sebaceous-cysts-in-dogs#:~:text=Sebaceous%20cysts%20are%20usually%20harmless,skin%20or%20just%20beneath%20it.")
        elif breed_label == "Saluki":
            tab1, tab2, tab3= st.tabs(["Anesthetic idiosyncracy", "Hemolytic anemia", "Cataract"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Anesthetic idiosyncracy")
                    st.image("https://www.mdpi.com/animals/animals-14-00822/article_deploy/html/images/animals-14-00822-g001-550.jpg")
                with col2:
                    Anesthetic_idiosyncracy = '''A condition where an individual has an abnormal response to commonly used anesthetics sometimes leading to death. Idiosyncratic means there is no good explanation or way to predict this.'''
                    st.markdown(Anesthetic_idiosyncracy)
                with st.expander("See More Details"):
                    st.subheader("Symptoms")
                    st.write(" An abnormal, unreliable response to commonly used anaesthetics. In severe cases it can lead to cardiac and/or respiratory arrest during the surgical procedure with the danger of a fatal outcome. Unfortunately, this reaction is completely unpredictable and there is no certain way to predict or determine this kind of response.")
                    st.markdown("---")
                    st.subheader("Disease Cause")
                    st.write("It is believed to be caused by the incapability of the liver to properly metabolise anaesthetic agents.")
                    st.markdown("---")
                    st.link_button("Source","https://ngdc.cncb.ac.cn/idog/disease/getDiseaseDetailById.action?diseaseId=14")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hemolytic anemia")
                    st.image("https://eclinpath.com/wp-content/uploads/immune-mediated-hemolytic-anemia-in-a-dog.jpg")
                with col2:
                    Hemolytic_anemia = '''A condition in which affected dogs’ immune system fights and destroys typically healthy red blood cells. This condition can be a primary condition or be a result of a secondary, or underlying illness.'''
                    st.markdown(Hemolytic_anemia)
                with st.expander("See More Details"):
                    st.subheader("Causes of Hemolytic Anemia in Dogs")
                    st.write("Autoimmune hemolytic anemia in dogs may be either primary or secondary. This autoimmune disease destroys the red blood cells as they circulate within the spleen or liver. The liver becomes overworked, as it must rid itself of the overproduced hemoglobin. ")
                    st.markdown("---")
                    st.subheader("Diagnosis of Hemolytic Anemia in Dogs")
                    st.write("If you suspect your dog is suffering from anemia, make an appointment with your veterinarian so he can be assessed. The veterinarian will perform a complete physical examination, and will focus on a test called a complete blood count, or CBC. The complete blood count test accurately will measure several variables within one sample of blood. These include the amount and percentage of red blood cells within the sample. Once the sample is taken and measured, the medical professional will take a closer look at the shape and size of the cells to check for abnormalities in both categories. In hemolytic anemia, the shape, as well as the size, will be atypical. They may also be abnormally clumped together, known as autoagglutination. ")
                    st.write("Once the CBC comes back as anemia, the veterinarian will want to do testing to see what is specifically causing it, and to see if the anemia is from a primary cause, or secondary.  Further evaluations may include serologic blood tests to check for parasites, a Coombs test to check for antibodies, and lab testing for finding the specific number or percentage of immature blood cells, known as reticulocytes. ")
                    st.write("Other testing will continue, especially if the veterinarian suspects a secondary cause for your dog’s hemolytic anemia. The veterinarian may perform a biochemistry profile to check for the functionality of his organs, urinalysis to check for kidney function and for any urinary tract infection, chest x-rays to test for cancer within the lungs, abdominal x-rays to test for cancer, as well as an abdominal ultrasound.")
                    st.markdown("---")
                    st.subheader("Treatment of Hemolytic Anemia in Dogs")
                    st.write("A **blood transfusion** may need to be performed if your dog’s anemia is severe. Samples of blood will be drawn for baseline testing, and the blood transfusion will be performed to keep your dog stable while the specific cause of the anemia is diagnosed.")
                    st.write("**Immunosuppressive therapy** will be suggested if the hemolytic anemia is determined to be of primary origin. The veterinarian may choose to administer doses of corticosteroid medications or other immunosuppressive medications recommended by the medical professional.")
                    st.write("If your dog’s hemolytic anemia is the cause of a specific underlying disorder or disease, the treatment will depend on the disease he is suffering from. Once your dog is diagnosed with a specific disease causing the blood disorder, your veterinarian will discuss with you treatment options. Once the secondary disease is treated, your dog’s hemolytic anemia will subside in time.")
                    st.markdown("---")
                    st.link_button("Source","https://wagwalking.com/condition/hemolytic-anemia")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/")          
        elif breed_label == "Scottish Deerhound":
            tab1, tab2, tab3= st.tabs(["Bloat", "Gastric torsion", "Cataract"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Bloat")
                    st.image("https://www.akc.org/wp-content/uploads/2021/09/Senior-Beagle-lying-on-a-rug-indoors.jpg")
                with col2:
                    Bloat = '''Bloat, also known as gastric dilatation-volvulus (GDV) complex, is a medical and surgical emergency. As the stomach fills with air, pressure builds, stopping blood from the hind legs and abdomen from returning to the heart. Blood pools at the back end of the body, reducing the working blood volume and sending the dog into shock.'''
                    st.markdown(Bloat)
                with st.expander("See More Details"):
                    st.subheader("What Are the Signs of Bloat in Dogs?")
                    st.write("An enlargement of the dog’s abdomen")
                    st.write("Retching")
                    st.write("Salivation")
                    st.write("Restlessness")
                    st.write("An affected dog will feel pain and might whine if you press on his belly")
                    st.write("Without treatment, in only an hour or two, your dog will likely go into shock. The heart rate will rise and the pulse will get weaker, leading to death.")
                    st.markdown("---")
                    st.subheader("Why Do Dogs Bloat?")
                    st.write("This question has perplexed veterinarians since they first identified the disease. We know air accumulates in the stomach (dilatation), and the stomach twists (the volvulus part). We don’t know if the air builds up and causes the twist, or if the stomach twists and then the air builds up.")
                    st.markdown("---")
                    st.subheader("How Is Bloat Treated?")
                    st.write("Veterinarians start by treating the shock. Once the dog is stable, he’s taken into surgery. We do two procedures. One is to deflate the stomach and turn it back to its correct position. If the stomach wall is damaged, that piece is removed. Second, because up to 90 percent of affected dogs will have this condition again, we tack the stomach to the abdominal wall (a procedure called a gastropexy) to prevent it from twisting.")
                    st.markdown("---")
                    st.subheader("How Can Bloat Be Prevented?")
                    st.write("If a dog has relatives (parents, siblings, or offspring) who have suffered from bloat, there is a higher chance he will develop bloat. These dogs should not be used for breeding.")
                    st.write("Risk of bloat is correlated to chest conformation. Dogs with a deep, narrow chest — very tall, rather than wide — suffer the most often from bloat. Great Danes, who have a high height-to-width ratio, are five-to-eight times more likely to bloat than dogs with a low height-to-width ratio.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/bloat-in-dogs/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Gastric torsion")
                    st.image("https://www.mcahonline.com/wp-content/uploads/2020/06/MCAH_6.22.2020-300x200.jpg")
                with col2:
                    Gastric_torsion = '''Also known as bloat, a twisted stomach, or Gastric Dilation-Volvulus (GDV). There are two parts to this condition. The first part is the bloating where a dog’s stomach fills up with gas, fluid, food, or any combination of the three massively. Torsion or volvulus is the second part where the entire stomach twists around itself inside of the abdomen. As a result, the abdomen closes off at both the entrance and exit. Today we’re going to go over the causes of gastric torsion, the signs/symptoms, and treatment. '''
                    st.markdown(Gastric_torsion)
                with st.expander("See More Details"):
                    st.subheader("What causes gastric torsion?")
                    st.write("Veterinarians aren’t sure about the exact cause of bloat or torsion. However, we believe some factors can put dogs at a higher risk. These factors include:")
                    st.write("Your pet eating from a food bowl that’s too high")
                    st.write("A dog eating only one big meal a day")
                    st.write("A dog eating too quickly")
                    st.write("A dog running or playing right after they eat")
                    st.write("Genetics")
                    st.write("Overeating/drinking")
                    st.write("If your dog is a part of a bigger breed, esp. if they have deep chests.")
                    st.write("If your dog is older (especially older than seven years old)")
                    st.write("Male dogs also experience gastric torsion more often than female dogs")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("If your canine companion shows all of the classic signs of gastric torsion, this is a life-threatening condition and you need to bring your pet to a veterinarian immediately. One of our experienced veterinarians will evaluate your dog and take an x-ray for confirmation. If your dog is experiencing shock, we will place an IV catheter to administer fluids and medications to stabilize your pet prior to surgery. ")
                    st.write("Treatments vary depending on the severity of the condition. If your pet is not experiencing torsion, a veterinarian can put a tube down the throat to release any built-up pressure. A twisted stomach (determined via X-ray) can stop the tube from entering the throat. In the case of a twisted stomach, emergency surgery will need to happen. Aside from surgery, your dog will need continued fluids through an IV and medications. We will also continue to monitor your dog’s heart for any signs of abnormalities that can be a side effect of gastric torsion.")
                    st.write("The good news is that as with any condition, the earlier you detect the signs, the better. Prevention involves making sure that your fur baby is eating at eye level, not playing right away after mealtime, and making sure that they’re eating well-balanced meals (2-3 small meals a day). For those high-risk breeds, we can also perform a surgery called a gastropexy. This is when one of our veterinarians tacks the stomach to the body wall, greatly reducing the likelihood that it can twist. If you have more questions about gastric torsion or gastropexy, call to talk to one of our amazing veterinarians!")
                    st.markdown("---")
                    st.link_button("Source","https://www.mcahonline.com/gastric-torsion-in-dogs/")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/")
        elif breed_label == "Weimaraner":
            tab1, tab2, tab3= st.tabs(["Bloat", "Demodicosis", "Distichiasis"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Bloat")
                    st.image("https://www.akc.org/wp-content/uploads/2021/09/Senior-Beagle-lying-on-a-rug-indoors.jpg")
                with col2:
                    Bloat = '''Bloat, also known as gastric dilatation-volvulus (GDV) complex, is a medical and surgical emergency. As the stomach fills with air, pressure builds, stopping blood from the hind legs and abdomen from returning to the heart. Blood pools at the back end of the body, reducing the working blood volume and sending the dog into shock.'''
                    st.markdown(Bloat)
                with st.expander("See More Details"):
                    st.subheader("What Are the Signs of Bloat in Dogs?")
                    st.write("An enlargement of the dog’s abdomen")
                    st.write("Retching")
                    st.write("Salivation")
                    st.write("Restlessness")
                    st.write("An affected dog will feel pain and might whine if you press on his belly")
                    st.write("Without treatment, in only an hour or two, your dog will likely go into shock. The heart rate will rise and the pulse will get weaker, leading to death.")
                    st.markdown("---")
                    st.subheader("Why Do Dogs Bloat?")
                    st.write("This question has perplexed veterinarians since they first identified the disease. We know air accumulates in the stomach (dilatation), and the stomach twists (the volvulus part). We don’t know if the air builds up and causes the twist, or if the stomach twists and then the air builds up.")
                    st.markdown("---")
                    st.subheader("How Is Bloat Treated?")
                    st.write("Veterinarians start by treating the shock. Once the dog is stable, he’s taken into surgery. We do two procedures. One is to deflate the stomach and turn it back to its correct position. If the stomach wall is damaged, that piece is removed. Second, because up to 90 percent of affected dogs will have this condition again, we tack the stomach to the abdominal wall (a procedure called a gastropexy) to prevent it from twisting.")
                    st.markdown("---")
                    st.subheader("How Can Bloat Be Prevented?")
                    st.write("If a dog has relatives (parents, siblings, or offspring) who have suffered from bloat, there is a higher chance he will develop bloat. These dogs should not be used for breeding.")
                    st.write("Risk of bloat is correlated to chest conformation. Dogs with a deep, narrow chest — very tall, rather than wide — suffer the most often from bloat. Great Danes, who have a high height-to-width ratio, are five-to-eight times more likely to bloat than dogs with a low height-to-width ratio.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/bloat-in-dogs/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Demodicosis")
                    st.image("https://images.wagwalkingweb.com/media/articles/dog/demodex-/demodex-.jpg?auto=compress&fit=max&width=640")
                with col2:
                    Demodicosis = '''are cigar shaped microscopic parasitic mites that live within the hair follicles of all dogs. These mites are passed to puppies from their mothers in the first few days of life, and then live within the hair follicles for the duration of animal’s life without causing problems'''
                    st.markdown(Demodicosis)
                with st.expander("See More Details"):
                    st.subheader("What causes demodicosis?")
                    st.write("There are two presentations of demodicosis depending on the age at which it develops. Juvenile onset demodicosis tends to occur in puppyhood between the ages of 3 months and 18 months, and occurs in both localised and generalised forms. The exact cause is quite poorly understood but probably occurs due to a mite specific genetic defect in the immune system which allows mite numbers to increase. This defect may or may not resolve as the puppy ages. It is thought to be ‘mite specific’ because these puppies are healthy in all other respects and do not succumb to other infections. Generalised demodicosis can be a very severe disease. Adult onset demodicosis usually occurs in the generalised form and in dogs over 4 years of age. It is generally considered a more severe disease than its juvenile onset counterpart. In these cases, mite numbers have been controlled in normal numbers in the hair follicles for years prior to the onset of disease, which tends to result from a systemic illness affecting the immune system. Common triggers for adult onset demodicosis include hormonal diseases and cancer.")
                    st.markdown("---")
                    st.subheader("What are the clinical signs?")
                    st.write("Localised demodicosis in juvenile dogs presents as patches of hair loss and red inflamed skin. These patches often occur around the face, head and feet and are not typically itchy")
                    st.markdown("---")
                    st.subheader("How is it diagnosed?")
                    st.write("Demodicosis can often be suspected following a review of the animal’s history and assessment of the clinical signs. The parasitic mites within the hair follicles result in plugging and the formation of ‘black heads’. The plugged follicles also cause large amounts of scale to be present on the hairs themselves.")
                    st.write("Demodicosis can usually be diagnosed relatively easily. Hairs can be plucked from the affected skin and then examined under a microscope for the presence of the mites. Alternatively, the skin can be squeezed and then scraped with a blade to collect up the surface debris from the skin. This material is then also examined under a microscope for the parasites.")
                    st.markdown("---")
                    st.subheader("Is it contagious?")
                    st.write("Demodex mites from dogs are considered non-infectious to in-contact animals and people. It is thought that Demodex mites can only be passed between dogs in the first few days of life from the mother to the pup.")
                    st.markdown("---")
                    st.subheader("How is it treated?")
                    st.write("The treatment used for demodicosis depends on the age of the animal and the severity of the disease. Mild and localised forms of demodicosis in young dogs may not require treatment, and may resolve spontaneously as the animal ages. These cases should be closely monitored if no treatment is given.")
                    st.write("Generalised cases in young dogs and those in adult dogs require intensive treatment. Secondary infections must be treated with courses of antibiotics, and a swab is often submitted to a laboratory to grow the organisms to ensure the correct antibiotic is selected. The licensed treatments for demodicosis in the UK include a dip solution called Aludex and a spot-on product called Advocate. The dip is performed on a weekly basis until mite numbers are brought under control. Advocate spot-on is generally used for milder cases and is usually used monthly. In severe cases not responding to the licensed treatments, off-licence treatments must be used. Some of these drugs, such as Ivermectin and Milbemycin, are used for demodicosis in other countries.")
                    st.markdown("---")
                    st.subheader("What is the prognosis?")
                    st.write("The prognosis for localised disease in young dogs is very good, and most recover uneventfully from the disease. Generalised cases in young dogs can take many weeks or even months of treatment, but it is usually possible to control the disease with a good long term outlook.")
                    st.write("The prognosis for adult onset generalised demodicosis is far more uncertain, as many of these dogs have an underlying systemic illness. If this illness can be identified and cured, the prognosis for managing the demodicosis is much better. Some cases require long term medication to keep mite numbers controlled.")
                    st.markdown("---")
                    st.link_button("Source","https://www.ndsr.co.uk/information-sheets/canine-demodicosis/#:~:text=And%20what%20is%20demodicosis%3F,animal's%20life%20without%20causing%20problems.")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Distichiasis")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/2136//2023_2135i_distichia_eye_6021.jpg")
                with col2:
                    Distichiasis = '''A distichia (plural distichiae) is an extra eyelash that grows from the margin of the eyelid through the duct or opening of the meibomian gland or adjacent to it. Meibomian glands produce lubricants for the eye and their openings are located along the inside edge of the eyelids. The condition in which these abnormal eyelashes are found is called distichiasis.'''
                    st.markdown(Distichiasis)
                with st.expander("See More Details"):
                    st.subheader("What causes distichiasis?")
                    st.write("Sometimes eyelashes arise from the meibomian glands. Why the follicles develop in this abnormal location is not known, but the condition is recognized as a hereditary problem in certain breeds of dogs. Distichiasis is a rare disorder in cats.")
                    st.markdown("---")
                    st.subheader("What breeds are more likely to have distichiasis?")
                    st.write("The more commonly affected breeds include the American Cocker Spaniel, Cavalier King Charles Spaniel, Shih Tzu, Lhasa Apso, Dachshund, Shetland Sheepdog, Golden Retriever, Chesapeake Retriever, Bulldog, Boston Terrier, Pug, Boxer Dog, Maltese, and Pekingese.")
                    st.markdown("---")
                    st.subheader("How is distichiasis diagnosed?")
                    st.write("Distichiasis is usually diagnosed by identifying lashes emerging from the meibomian gland openings or by observing lashes that touch the cornea or the conjunctival lining of the affected eye. A thorough eye examination is usually necessary, including fluorescein staining of the cornea and assessment of tear production in the eyes, to assess the extent of any corneal injury and to rule out other causes of the dog's clinical signs. Some dogs will require topical anesthetics or sedatives to relieve the intense discomfort and allow a thorough examination of the tissues surrounding the eye.")
                    st.markdown("---")
                    st.subheader("How is the condition treated?")
                    st.write("Dogs that are not experiencing clinical signs with short, fine distichia may require no treatment at all. Patients with mild clinical signs may be managed conservatively, through the use of ophthalmic lubricants to protect the cornea and coat the lashes with a lubricant film. Removal of distichiae is no longer recommended, as they often grow back thicker or stiffer, but they may be removed for patients unable to undergo anesthesia or while waiting for a more permanent procedure.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/distichia-or-distichiasis-in-dogs")
        elif breed_label == "Staffordshire Bullterrier":
            tab1, tab2, tab3= st.tabs(["Cataract", "Epilepsy", "Hemangiosarcoma"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Epilepsy")
                    st.image("https://canna-pet.com/wp-content/uploads/2017/03/CP_EpilepsyDogs_1.jpg")
                with col2:
                    Epilepsy = '''A brain disorder characterized by recurrent seizures without a known cause or abnormal brain lesion (brain injury or disease). In other words, the brain appears to be normal but functions abnormally. A seizure is a sudden surge in the electrical activity of the brain causing signs such as twitching, shaking, tremors, convulsions, and/or spasms.'''
                    st.markdown(Epilepsy)
                with st.expander("See More Details"):
                    st.subheader("What Are the Symptoms of Seizures?")
                    st.write("Symptoms can include collapsing, jerking, stiffening, muscle twitching, loss of consciousness, drooling, chomping, tongue chewing, or foaming at the mouth. Dogs can fall to the side and make paddling motions with their legs. They sometimes poop or pee during the seizure. They are also not aware of their surroundings. Some dogs may look dazed, seem unsteady or confused, or stare off into space before a seizure. Afterward, your dog may be disoriented, wobbly, or temporarily blind. They may walk in circles and bump into things. They might have a lot of drool on their chin. They may try to hide.")
                    st.markdown("---")
                    st.subheader("How is epilepsy diagnosed?")
                    st.write("Epilepsy is a diagnosis of exclusion; the diagnosis of epilepsy is made only after all other causes of seizures have been ruled out. A thorough medical history and physical examination are performed, followed by diagnostic testing such as blood and urine tests and radiographs (X-rays). Additional tests such as bile acids, cerebrospinal fluid (CSF) testing, computed tomography (CT) or magnetic resonance imaging (MRI) may be recommended, depending on the initial test results. In many cases a cause is not found; these are termed idiopathic. Many epilepsy cases are grouped under this classification as the more advanced testing is often not carried out due to cost or availability. A dog’s age when seizures first start is also a prevalent factor in coming to a diagnosis.")
                    st.markdown("---")
                    st.subheader("What is the treatment of epilepsy?")
                    st.write("Anticonvulsants (anti-seizure medications) are the treatment of choice for epilepsy. There are several commonly used anticonvulsants, and once treatment is started, it will likely be continued for life. Stopping these medications suddenly can cause seizures.")
                    st.write("The risk and severity of future seizures may be worsened by stopping and re- starting anticonvulsant drugs. Therefore, anticonvulsant treatment is often only prescribed if one of the following criteria is met:")
                    st.write("**More than one seizure a month:** You will need to record the date, time, length, and severity of all episodes in order to determine medication necessity and response to treatment.")
                    st.write("**Clusters of seizures:** If your pet has groups or 'clusters' of seizures, (one seizure following another within a very short period of time), the condition may progress to status epilepticus, a life- threatening condition characterized by a constant, unending seizure that may last for hours. Status epilepticus is a medical emergency.")
                    st.write("**Grand mal or severe seizures:** Prolonged or extremely violent seizure episodes. These may worsen over time without treatment.")
                    st.markdown("---")
                    st.subheader("What is the prognosis for a pet with epilepsy?")
                    st.write("Most dogs do well on anti-seizure medication and are able to resume a normal lifestyle. Some patients continue to experience periodic break-through seizures. Many dogs require occasional medication adjustments, and some require the addition of other medications over time.")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/epilepsy-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hemangiosarcoma ")
                    st.image("https://www.mdpi.com/vetsci/vetsci-10-00387/article_deploy/html/images/vetsci-10-00387-g001.png")
                with col2:
                    Hemangiosarcoma = '''Hemangiosarcoma (HSA) is a highly invasive canine cancer. This cancer causes blood vessels to branch, fragment, become leaky, and ultimately rupture. It’s the cause of about two-thirds of heart and splenic tumors, with metastasis (secondary malignant growths) affecting the liver, lungs, lymph nodes, and bones. '''
                    st.markdown(Hemangiosarcoma )
                with st.expander("See More Details"):
                    st.subheader("Diagnosis")
                    st.write("There are no easy lab tests that can diagnose hemangiosarcoma. X-rays and ultrasounds will show the size and location of a mass but won’t definitively tell you if your dog has cancer. The only way to truly diagnose HSA is to surgically remove the affected tissue and send it to a pathologist. But doing that must be done by a veterinary surgeon, as it is complicated and possibly dangerous.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("When a dog is diagnosed with hemangiosarcoma, and it’s too late to treat it, studies show the dog is most likely to live only a few more weeks. “Survival times usually do not exceed one year, even with surgical and chemotherapeutic treatments,” says Sams. I lost Fin within three weeks.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hemangiosarcoma-in-dogs/#:~:text=Hemangiosarcoma%20(HSA)%20is%20a%20highly,%2C%20lymph%20nodes%2C%20and%20bones.")
        elif breed_label == "American Staffordshire Bullterrier":
            tab1, tab2, tab3= st.tabs(["Cataract", "Hypothyroidism", "Hip dysplasia"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hypothyroidism")
                    st.image("https://www.lifelearn-cliented.com//cms/resources/body/24023/2024_817i_thyroid_dog_5002.png")
                with col2:
                    Hypothyroidism = ''' A condition of inadequate thyroid hormone levels that leads to a reduction in a dog's metabolic state. Hypothyroidism is one of the most common hormonal (endocrine) diseases in dogs. It generally affects middle-aged dogs (average of 6–7 years of age), and it may be more common in spayed females and neutered males. A wide variety of breeds may be affected.'''
                    st.markdown(Hypothyroidism)
                with st.expander("See More Details"):
                    st.subheader("What causes hypothyroidism?")
                    st.write("In dogs, hypothyroidism is usually caused by one of two diseases: lymphocytic thyroiditis or idiopathic thyroid gland atrophy. **Lymphocytic thyroiditis** is the most common cause of hypothyroidism and is thought to be an immune-mediated disease, meaning that the immune system decides that the thyroid is abnormal or foreign and attacks it. It is unclear why this occurs; however, it is a heritable trait, so genetics plays a role. In **idiopathic thyroid gland atrophy**, normal thyroid tissue is replaced by fat tissue. This condition is also poorly understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of hypothyroidism?")
                    st.write("When the metabolic rate slows down, virtually every organ in the body is affected. Most dogs with hypothyroidism have one or more of the following signs:")
                    st.write("weight gain without an increase in appetite")
                    st.write("lethargy (tiredness) and lack of desire to exercise")
                    st.write("cold intolerance (gets cold easily)")
                    st.write("dry, dull hair with excessive shedding")
                    st.write("very thin to nearly bald hair coat")
                    st.write("increased dark pigmentation in the skin")
                    st.write("increased susceptibility and occurrence of skin and ear infections")
                    st.write("failure to re-grow hair after clipping or shaving")
                    st.write("high blood cholesterol")
                    st.write("slow heart rate")
                    st.markdown("---")
                    st.subheader("How is hypothyroidism diagnosed?")
                    st.write("The most common screening test is a total thyroxin (TT4) level. This is a measurement of the main thyroid hormone in a blood sample. A low level of TT4, along with the presence of clinical signs, is suggestive of hypothyroidism. Definitive diagnosis is made by performing a free T4 by equilibrium dialysis (free T4 by ED) or a thyroid panel that assesses the levels of multiple forms of thyroxin. If this test is low, then your dog has hypothyroidism. Some pets will have a low TT4 and normal free T4 by ED. These dogs do not have hypothyroidism. Additional tests may be necessary based on your pet's condition. See handout “Thyroid Hormone Testing in Dogs” for more information.")
                    st.markdown("---")
                    st.subheader("Can it be treated?")
                    st.write("Hypothyroidism is treatable but not curable. It is treated with oral administration of thyroid replacement hormone. This drug must be given for the rest of the dog's life. The most recommended treatment is oral synthetic thyroid hormone replacement called levothyroxine (brand names Thyro-Tabs® Canine, Synthroid®).")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hypothyroidism-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hip-dysplasia-in-dogs/")
        
        elif breed_label == "Bedlington Terrier":  
            tab1, tab2, tab3= st.tabs(["Cataract", "Ectropion", "Microphthalmia"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Ectropion")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/23450//2023_1008i_Eye_ectropion_cross-section_2020-01.jpg")
                with col2:
                    Ectropion = '''Ectropion is an abnormality of the eyelids in which the eyelid (usually the lower eyelid) “rolls” outward or is everted. This causes the lower eyelids to appear droopy.'''
                    st.markdown(Ectropion)
                with st.expander("See More Details"):
                    st.subheader("What are the clinical signs of ectropion?")
                    st.write("The clinical signs are a “sagging” or “outward rolling” lower eyelid. A thick, mucoid discharge often accumulates along the eyelid margin. The eye and conjunctiva may appear reddened or inflamed. The dog may rub or paw at the eye if it becomes uncomfortable. Tears may run down the dog’s face if the medial aspect of the eyelid (the area of the eyelid toward the nose) is affected. In many cases, pigment contained in the tear fluid will cause a brownish staining of the fur beneath the eyes.")
                    st.markdown("---")
                    st.subheader("How is ectropion diagnosed?")
                    st.write("Diagnosis is usually made on physical examination. If the dog is older, blood and urine tests may be performed to search for an underlying cause for the ectropion. Corneal staining will be performed to assess the cornea and to determine if any corneal ulceration is present. Muscle or nerve biopsies may be recommended if neuromuscular disease is suspected. Testing for hypothyroidism and for antibodies against certain muscle fibers may be done if looking for underlying causes.")
                    st.markdown("---")
                    st.subheader("How is ectropion treated?")
                    st.write("The treatment for mild ectropion generally consists of medical therapy, such as lubricating eye drops and ointments to prevent the cornea and conjunctiva from drying out. Ophthalmic antibiotics may be recommended if corneal ulcers develop because of ectropion. If the condition is severe, the eyelids can be shortened surgically.")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/eyelid-ectropion-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Microphthalmia")
                    st.image("https://images.wagwalkingweb.com/media/articles/dog/microphthalmia-and-ocular-dysgenesis/microphthalmia-and-ocular-dysgenesis.jpg?auto=compress&fit=max&width=640")
                with col2:
                    Microphthalmia = '''Usually diagnosed in young puppies, microphthalmia* literally means ‘small eye’. Due to a defect early in fetal development, the eye does not grow at the same rate as the rest of the head and therefore looks smaller than it should. Eyes may look a little sunken. Third eyelids can be prominent.'''
                    st.markdown(Microphthalmia)
                with st.expander("See More Details"):
                    st.subheader("SIGNS AND SYMPTOMS")
                    st.write("This condition is apparent in pups once their eyes have opened. Affected eyes are smaller than normal and appear recessed. The third eyelid may be more prominent. One or both eyes may be affected.")
                    st.markdown("---")
                    st.subheader("CAUSES")
                    st.write("This is a congenital condition (present at birth), being nondevelopment of the eye. In some breeds, it has been found to be inherited, often associated with “merle” coat coloring. The genetics have been well-studied in mice, where mutations in the regulatory gene MITF (microphthalmia-associated transcription factor) affect the development of pigment cells (melanocytes) and can lead to microphthalmia, deafness, and loss of pigmentation. The MITF gene in dogs has recently been identified and shown to be associated with white spotting or piebald coat coloring in several breeds. Microphthalmia might also be inherited in Samoyeds, but this has not been determined. There are also references to pesticides and wormers causing the problem.")
                    st.markdown("---")
                    st.subheader("DIAGNOSTIC TESTS")
                    st.write("It is visible to the naked eye. The third eyelid (the white bit in the corner) often looks larger than normal. Examination by a veterinary ophthalmologist, who will also examine your dog’s eyes thoroughly for other abnormalities, will give a definitive diagnosis. The examination is non-invasive, using light and magnifying lenses to examine and look inside the eye.")
                    st.markdown("---")
                    st.subheader("TREATMENT GUIDELINES")
                    st.write("The basic defect can’t be treated. Associated complications, such as glaucoma, are treated as required.")
                    st.markdown("---")
                    st.link_button("Source","https://www.samoyedhealthfoundation.org/diseases/microphthalmia/")
        elif breed_label == "Border Terrier":     
            tab1, tab2, tab3= st.tabs(["Cataract", "Cryptorchidism", "Lens luxation"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cryptorchidism")
                    st.image("https://iloveveterinary.com/wp-content/uploads/2021/05/Cryptorchid-Chihuahua.jpg.webp", width=200)
                with col2:
                    Cryptorchidism = '''the medical term that refers to the failure of one or both testicles (testes) to descend into the scrotum. The testes develop near the kidneys within the abdomen and normally descend into the scrotum by two months of age. In certain dogs, it may occur later, but rarely after six months of age. Cryptorchidism may be presumed to be present if the testicles cannot be felt in the scrotum after two to four months of age.
                            '''
                    st.markdown(Cryptorchidism)
                with st.expander("See More details"):
                    st.subheader("If the testicles aren't in the scrotum, where are they?")
                    st.write("In most cases of cryptorchidism, the testicle is retained in the abdomen or in the inguinal canal (the passage through the abdominal wall into the genital region through which a testicle normally descends). Sometimes, the testicle will be located in the subcutaneous tissues (just under the skin) in the groin region, between the inguinal canal and the scrotum.")
                    st.markdown("---")
                    st.subheader("How is cryptorchidism diagnosed?")
                    st.write("In cases of abdominal cryptorchidism, the testicle cannot be felt from the outside. An abdominal ultrasound or radiographs (X-rays) may be performed to determine the exact location of the retained testicle, but this is not often done before surgery, as it is not required to proceed with surgery. Typically, only one testicle is retained, and this is called unilateral cryptorchidism. If you have a dog that does not appear to have testicles but is exhibiting male behaviors, a hormonal test called an hCG stimulation test can be performed to see if he is already neutered.")
                    st.markdown("---")
                    st.subheader("What causes cryptorchidism and how common is it?")
                    st.write("Cryptorchidism occurs in all breeds but toy breeds, including toy Poodles, Pomeranians, and Yorkshire Terriers, may be at higher risk. Approximately 75% of cases of cryptorchidism involve only one retained testicle while the remaining 25% involve failure of both testicles to descend into the scrotum. The right testicle is more than twice as likely to be retained as the left testicle. Cryptorchidism affects approximately 1-3% of all dogs. The condition appears to be inherited since it is commonly seen in families of dogs, although the exact cause is not fully understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of cryptorchidism?")
                    st.write("This condition is rarely associated with pain or other signs unless a complication develops. In its early stages, a single retained testicle is significantly smaller than the other, normal testicle. If both testicles are retained, the dog may be infertile. The retained testicles continue to produce testosterone but generally fail to produce sperm.")
                    st.markdown("---")
                    st.subheader("What is the treatment for cryptorchidism?")
                    st.write("Neutering and removal of the retained testicle(s) are recommended. If only one testicle is retained, the dog will have two incisions - one for extraction of each testicle. If both testicles are in the inguinal canal, there will also be two incisions. If both testicles are in the abdomen, a single abdominal incision will allow access to both.")
                    st.markdown("---")
                    st.subheader("What if I don't want to neuter my dog?")
                    st.write("There are several good reasons for neutering a dog with cryptorchidism. The first reason is to remove the genetic defect from the breed line. Cryptorchid dogs should never be bred. Second, dogs with a retained testicle are more likely to develop a testicular tumor (cancer) in the retained testicle. Third, as described above, the testicle can twist, causing pain and requiring emergency surgery to correct. Finally, dogs with a retained testicle typically develop the undesirable characteristics associated with intact males like urine marking and aggression. The risk of developing testicular cancer is estimated to be at least ten times greater in dogs with cryptorchidism than in normal dogs.")
                    st.markdown("---")
                    st.subheader("What is the prognosis for a dog with cryptorchidism?")
                    st.write("The prognosis is excellent for dogs that undergo surgery early before problems develop in the retained testicle. The surgery is relatively routine, and the outcomes are overwhelmingly positive.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/retained-testicle-cryptorchidism-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:  
                    st.header("Lens luxation")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Lens_luxation = '''A painful and potentially blinding inherited canine eye condition. Lens luxation occurs when the ligaments supporting the lens weaken, displacing it from its normal position. Signs of lens luxation may include red, teary, hazy, or cloudy, painful eyes. PLL can cause eye inflammation and glaucoma, particularly if the lens shifts forward into the eye. '''
                    st.markdown(Lens_luxation)
                with st.expander("See More Details"):
                    st.subheader("Causes")
                    st.write("The lens is a structure in the eye located behind the iris (the colored portion of the eye) responsible for focusing light onto the retina for visualization. It is suspended in the eye by multiple ligaments called zonules. PLL is caused by an inherited weakness and breakdown of the zonules, displacing the lens from its normal position in the eye. The direction that the lens luxates can be either forward (anterior) or backward (posterior). Anterior lens luxation is the most damaging and considered an emergency as it can rapidly increase pressure inside the eye, known as glaucoma, causing pain and potentially blindness. Posterior lens luxation leads to milder inflammation, and glaucoma is less likely to develop.")
                    st.write("PLL most commonly develops in dogs between the ages of three and eight. However, structural changes in the eye may already be evident at 20 months of age, long before lens luxation typically occurs. Both eyes are often affected by PLL, but not necessarily at the same time. This differs from secondary lens luxation, which can more commonly only affect one eye and is usually caused by a coexisting ocular disease such as glaucoma, inflammatory conditions of the eye (uveitis), cataracts, eye trauma and eye tumors.")
                    st.markdown("---")
                    st.subheader("Diagnosis")
                    st.write("Early detection of lens luxation is crucial. Your veterinarian will diagnose primary lens luxation by performing a complete eye exam. They may measure your dog’s eye pressure for secondary conditions like glaucoma. You may be referred to a veterinary ophthalmology specialist where additional testing could include an eye ultrasound to evaluate the internal structures of the eye.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("Treatment options vary by stage of disease and position of the lens. When diagnosed early, the most common treatment for anterior lens luxation is surgery to remove the lens by a veterinarian specializing in ophthalmology. Topical eye medications may be needed long-term, even after surgery.")
                    st.write("If glaucoma develops suddenly, this requires emergency management and may include medication to decrease eye pressure, followed by referral to a veterinary ophthalmologist. If the eye has uncontrolled glaucoma, is permanently blind, or there is pain or inflammation, it may be necessary for the affected eye to be surgically removed (enucleation).")
                    st.write("Treatment for posterior lens luxation may include topical medications to help prevent the lens from shifting forward and causing more severe damage to the eye.")
                    st.markdown("---")
                    st.subheader("Outcome")
                    st.write("Primary lens luxation most commonly progresses to affect both eyes. For this reason, regular and in-depth ocular examinations are recommended in at-risk dogs. Anterior lens luxation left untreated or not addressed immediately often has a poor prognosis for saving the eye.")
                    st.write("Dogs that receive surgery early for anterior lens luxation can often preserve some vision but may have diminished vision that is more blurred up close. However, this doesn’t generally appear to affect everyday life. Surgery is not without risk of complications, and often, patients require lifelong topical eye medications.")
                    st.markdown("---")
                    st.link_button("Source","https://www.vet.cornell.edu/departments/riney-canine-health-center/canine-health-information/primary-lens-luxation#:~:text=Lens%20luxation%20occurs%20when%20the,shifts%20forward%20into%20the%20eye.")
        elif breed_label == "Kerry Blue Terrier":  
            tab1, tab2, tab3= st.tabs(["Cataract", "Distichiasis", "Entropion"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Distichiasis")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/2136//2023_2135i_distichia_eye_6021.jpg")
                with col2:
                    Distichiasis = '''A distichia (plural distichiae) is an extra eyelash that grows from the margin of the eyelid through the duct or opening of the meibomian gland or adjacent to it. Meibomian glands produce lubricants for the eye and their openings are located along the inside edge of the eyelids. The condition in which these abnormal eyelashes are found is called distichiasis.'''
                    st.markdown(Distichiasis)
                with st.expander("See More Details"):
                    st.subheader("What causes distichiasis?")
                    st.write("Sometimes eyelashes arise from the meibomian glands. Why the follicles develop in this abnormal location is not known, but the condition is recognized as a hereditary problem in certain breeds of dogs. Distichiasis is a rare disorder in cats.")
                    st.markdown("---")
                    st.subheader("What breeds are more likely to have distichiasis?")
                    st.write("The more commonly affected breeds include the American Cocker Spaniel, Cavalier King Charles Spaniel, Shih Tzu, Lhasa Apso, Dachshund, Shetland Sheepdog, Golden Retriever, Chesapeake Retriever, Bulldog, Boston Terrier, Pug, Boxer Dog, Maltese, and Pekingese.")
                    st.markdown("---")
                    st.subheader("How is distichiasis diagnosed?")
                    st.write("Distichiasis is usually diagnosed by identifying lashes emerging from the meibomian gland openings or by observing lashes that touch the cornea or the conjunctival lining of the affected eye. A thorough eye examination is usually necessary, including fluorescein staining of the cornea and assessment of tear production in the eyes, to assess the extent of any corneal injury and to rule out other causes of the dog's clinical signs. Some dogs will require topical anesthetics or sedatives to relieve the intense discomfort and allow a thorough examination of the tissues surrounding the eye.")
                    st.markdown("---")
                    st.subheader("How is the condition treated?")
                    st.write("Dogs that are not experiencing clinical signs with short, fine distichia may require no treatment at all. Patients with mild clinical signs may be managed conservatively, through the use of ophthalmic lubricants to protect the cornea and coat the lashes with a lubricant film. Removal of distichiae is no longer recommended, as they often grow back thicker or stiffer, but they may be removed for patients unable to undergo anesthesia or while waiting for a more permanent procedure.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/distichia-or-distichiasis-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Entropion")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Entroption = '''when the eyelids roll inward toward the eye. The fur on the eyelids and the eyelashes then rub against the surface of the eye (the cornea). This is a very painful condition that can lead to corneal ulcers.'''
                    st.markdown(Entroption)
                with st.expander("See More Details"):
                    st.write("Many Bloodhounds have abnormally large eyelids (macroblepharon) which results in an unusually large space between the eyelids.  Because of their excessive facial skin and resulting facial droop, there is commonly poor support of the outer corner of the eyelids")
                    st.markdown("---")
                    st.subheader("How is entropion treated?")
                    st.write("The treatment for entropion is surgical correction. A section of skin is removed from the affected eyelid to reverse its inward rolling. In many cases, a primary, major surgical correction will be performed, and will be followed by a second, minor corrective surgery later. Two surgeries are often performed to reduce the risk of over-correcting the entropion, resulting in an outward-rolling eyelid known as ectropion. Most dogs will not undergo surgery until they have reached their adult size at six to twelve months of age.")
                    st.markdown("---")
                    st.subheader("Should an affected dog be bred?")
                    st.write("Due to the concern of this condition being inherited, dogs with severe ectropion requiring surgical correction should not be bred.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/eyelid-entropion-in-dogs")
        elif breed_label == "Irish Terrier":  
            tab1, tab2, tab3= st.tabs(["Cystinuria", "Digital hyperkeratosis", "Cardiac valvular disease"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cystinuria")
                    st.image("https://www.tambuzi.com/wp-content/uploads/2014/06/kidneystones.jpg")
                with col2:
                    Cystinuria = ''' an uncommon, inherited condition that causes an amino acid called cystine to build up in urine. Cystine can be excreted in urine and lead to the formation of bladder or kidney stones.'''
                    st.markdown(Cystinuria)
                with st.expander("See More Details"):
                    st.subheader("Causes")
                    st.write("Cystine is a type of amino acid in the body that is normally reabsorbed by the kidneys. Cystinuria occurs when the kidneys are not able to properly reabsorb cystine, causing it to accumulate in the urine and form bladder or kidney stones.")
                    st.write("There are three types of cystinuria, two of which can occur in males and females, and one that is influenced by the presence of sex hormones common in intact males. ")
                    st.write("While both males and females with cystinuria are equally affected from excess cystine in their urine, the obstruction of urine flow is more common in males due to differences in their anatomy.")
                    st.markdown("---")
                    st.subheader("Diagnosis")
                    st.write("A test called a urinalysis will be performed to look for the presence of cystine crystals, the pH of the urine and any coexisting issues, such as a urinary tract infection. Cystine crystals form in acidic urine (which has a lower pH). ")
                    st.write("Another urine test called urine nitroprusside can screen for cystinuria. Specialized X-rays, ultrasounds or other urinary imaging tests may be used to diagnose cystine stones in the bladder or kidneys.  ")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("The goal of treatment is to reduce the amount of cystine excreted in the dog’s urine, dissolve what cystine remains and avoid stone formation. ")
                    st.write("Cystine can dissolve if the urine is made less acidic (by increasing its pH). This is achieved by feeding a prescription diet with reduced sodium and protein — particularly targeting an amino acid called methionine, which is one of the precursors involved in forming cystine stones.")
                    st.write("The urine should become diluted by feeding your dog a canned diet, adding water to their meals and encouraging them to drink more water. Medications may be needed if diet alone does not increase the urine pH enough to dissolve the cystine.")
                    st.write("After bladder stones develop, it is necessary to remove the stones and manage any secondary urinary tract infections or additional irritation. Removal often requires surgery. Neutering intact male dogs may be curative for certain types of androgen-dependent (sex hormone) cystinuria.   ")
                    st.write("Genetic testing is available for a few breeds known to be affected by cystinuria. And since cystinuria can be inherited, dogs suspected of having (or carrying) cystinuria should not be used for breeding without genetic testing and careful consideration of mate selection. ")
                    st.markdown("---")
                    st.link_button("Source","https://www.vet.cornell.edu/departments-centers-and-institutes/riney-canine-health-center/health-info/cystinuria#:~:text=Cystinuria%20is%20an%20uncommon%2C%20inherited,of%20bladder%20or%20kidney%20stones.")
            with tab2:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Digital hyperkeratosis")
                    st.image("https://www.tambuzi.com/wp-content/uploads/2014/06/kidneystones.jpg")
                with col2:
                    Digital_hyperkeratosis = '''A condition where the skin cells on the dog’s nose and paw pads grow excessively and fail to shed properly. This abnormal process results in an accumulation of keratin, a key skin and hair component, specifically in these areas.'''
                    st.markdown(Digital_hyperkeratosis)
                with st.expander("See More Details"):
                    st.subheader("What are the causes of hyperkeratosis?")
                    st.write("Hyperkeratosis can develop in dogs for a variety of reasons, and understanding these can help you better manage your pet's health. Here are some of the main causes and risk factors:")
                    st.write("**Congenital:** In some cases, hyperkeratosis is congenital, meaning dogs are born with this condition. This is often related to their genetic makeup and is more common in certain dog breeds. Cocker spaniels, boxers, English and French bulldogs, Boston terriers, beagles, and Basset hounds are predisposed.")
                    st.write("**Aging:** Older dogs are prone to hyperkeratosis. With age, the characteristics of this condition can change, often becoming more pronounced on the top of the dog’s nose and around the edges of the paw pads.")
                    st.write("**Systemic and Dermatologic Disorders:** Certain health conditions can lead to hyperkeratosis. For example, canine distemper virus infection can result in hyperkeratosis.")
                    st.write("**Idiopathic Disease:** Sometimes, hyperkeratosis occurs without any clear underlying cause, such as immune-mediated, infectious, hereditary, or anatomical issues. This is known as idiopathic hyperkeratosis, and its exact cause is not well understood.")
                    st.markdown("---")
                    st.subheader("Symptoms of hyperkeratosis in dogs")
                    st.write("The treatment options for hyperkeratosis in dogs will depend on the underlying cause and the severity of the condition. Here are some common methods used to manage this condition:")
                    st.write("**Topical Applications:** Topical treatments like ointments, creams, or balms can help soften and moisturize the affected areas. These products may contain ingredients like salicylic acid, urea, or propylene glycol to help break down excessive keratin and improve skin hydration.")
                    st.write("**Alternative Topical Treatments:** Other options include petroleum jelly, 50% propylene glycol, salicylic acid, ichthammol ointment, or tretinoin gel. These can help soften the affected areas and facilitate keratin removal.")
                    st.write("**Dietary Supplements:** Fatty acid supplements (omega fatty acids - fish oil) are often recommended to help improve skin and coat health in dogs with hyperkeratosis.")
                    st.write("**Paw Soaking:** Soaking your dog’s feet in warm water and/or propylene glycol can help soften the skin, making it easier to remove excess keratin. This also provides relief for sore or painful areas.")
                    st.write("**Antimicrobial and Steroid Ointments:** Ointments containing antimicrobials and steroids can be effective.")
                    st.write("**Systemic Antibiotics:** For severe infections, systemic antibiotics may be necessary. Your veterinarian may perform a skin culture and sensitivity test to choose the correct antibiotic.")
                    st.markdown("---")
                    st.subheader("Can hyperkeratosis be cured in dogs?")
                    st.write("Unfortunately, there is no cure for hyperkeratosis in dogs. It can be managed and treated, but it may require lifelong care and monitoring.")
                    st.markdown("---")
                    st.link_button("Source","https://www.kingsdale.com/hyperkeratosis-in-dogs-symptoms-causes-and-treatments")
            with tab3:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cardiac valvular disease")
                    st.image("https://www.tambuzi.com/wp-content/uploads/2014/06/kidneystones.jpg")
                with col2:
                    Cardiac_valvular_disease = '''Chronic degenerative valve disease (CVD) has many other names, such as endocardiosis, valvular regurgitation, valvular insufficiency, chronic valve disease, or myxomatous degeneration of the valve. This disease is a consequence of degeneration of the valves between the atrium and ventricle on both the right (tricuspid valve) and left (mitral valve) side of the heart, but the valve on the left side (mitral valve) is typically most severely affected. The degeneration causes the valves to become abnormally thick and develop a nodular “lumpy” appearance. This process is not caused by infection.'''
                    st.markdown(Cardiac_valvular_disease)
                with st.expander("See More Details"):
                    st.subheader("Who develops chronic degenerative valve disease?")
                    st.write("Chronic degenerative valve disease represents approximately 75% of all heart disease in dogs. Approximately 60% of affected dogs have degeneration of the mitral valve, 30% have degeneration of both the tricuspid and mitral valves, and the remaining 10% have degeneration in the tricuspid valve only.")
                    st.write("The risk of developing CVD increases as dogs get older and is rare before the age of 4 years. In addition, small breed dogs (dogs weighing less than 40 lb or (18.2 kg)) are more likely to get CVD than larger dogs. Certain breeds also have a higher risk of developing CVD.")
                    st.markdown("---")
                    st.subheader("What causes CVD?")
                    st.write("CVD is a degenerative process associated with aging. It is not caused by infection. Many older dogs with dental disease also have CVD, but the dental disease is not the cause of the CVD (though good dental hygiene is an important part of ensuring that your dog lives a long and healthy life). In addition, there is likely an inherited genetic component in some breeds, such as the Cavalier King Charles spaniel, but the genetic evidence for the majority of cases of CVD is lacking.")
                    st.markdown("---")
                    st.subheader("How is CVD treated in dogs?")
                    st.write("Medication is not required if the recommended tests determine that the heart is not enlarged and the blood pressure is normal (Stage B1). If the tests detect heart enlargement, and/or high blood pressure (Stage B2 or C), medication(s) may be prescribed.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/chronic-degenerative-valve-disease-in-dogs-in-depth")
        
        elif breed_label == "Norflok Terrier":  
            tab1, tab2, tab3= st.tabs(["Epilepsy", "Hypothyroidism", "Syncope"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Epilepsy")
                    st.image("https://canna-pet.com/wp-content/uploads/2017/03/CP_EpilepsyDogs_1.jpg")
                with col2:
                    Epilepsy = '''A brain disorder characterized by recurrent seizures without a known cause or abnormal brain lesion (brain injury or disease). In other words, the brain appears to be normal but functions abnormally. A seizure is a sudden surge in the electrical activity of the brain causing signs such as twitching, shaking, tremors, convulsions, and/or spasms.'''
                    st.markdown(Epilepsy)
                with st.expander("See More Details"):
                    st.subheader("What Are the Symptoms of Seizures?")
                    st.write("Symptoms can include collapsing, jerking, stiffening, muscle twitching, loss of consciousness, drooling, chomping, tongue chewing, or foaming at the mouth. Dogs can fall to the side and make paddling motions with their legs. They sometimes poop or pee during the seizure. They are also not aware of their surroundings. Some dogs may look dazed, seem unsteady or confused, or stare off into space before a seizure. Afterward, your dog may be disoriented, wobbly, or temporarily blind. They may walk in circles and bump into things. They might have a lot of drool on their chin. They may try to hide.")
                    st.markdown("---")
                    st.subheader("How is epilepsy diagnosed?")
                    st.write("Epilepsy is a diagnosis of exclusion; the diagnosis of epilepsy is made only after all other causes of seizures have been ruled out. A thorough medical history and physical examination are performed, followed by diagnostic testing such as blood and urine tests and radiographs (X-rays). Additional tests such as bile acids, cerebrospinal fluid (CSF) testing, computed tomography (CT) or magnetic resonance imaging (MRI) may be recommended, depending on the initial test results. In many cases a cause is not found; these are termed idiopathic. Many epilepsy cases are grouped under this classification as the more advanced testing is often not carried out due to cost or availability. A dog’s age when seizures first start is also a prevalent factor in coming to a diagnosis.")
                    st.markdown("---")
                    st.subheader("What is the treatment of epilepsy?")
                    st.write("Anticonvulsants (anti-seizure medications) are the treatment of choice for epilepsy. There are several commonly used anticonvulsants, and once treatment is started, it will likely be continued for life. Stopping these medications suddenly can cause seizures.")
                    st.write("The risk and severity of future seizures may be worsened by stopping and re- starting anticonvulsant drugs. Therefore, anticonvulsant treatment is often only prescribed if one of the following criteria is met:")
                    st.write("**More than one seizure a month:** You will need to record the date, time, length, and severity of all episodes in order to determine medication necessity and response to treatment.")
                    st.write("**Clusters of seizures:** If your pet has groups or 'clusters' of seizures, (one seizure following another within a very short period of time), the condition may progress to status epilepticus, a life- threatening condition characterized by a constant, unending seizure that may last for hours. Status epilepticus is a medical emergency.")
                    st.write("**Grand mal or severe seizures:** Prolonged or extremely violent seizure episodes. These may worsen over time without treatment.")
                    st.markdown("---")
                    st.subheader("What is the prognosis for a pet with epilepsy?")
                    st.write("Most dogs do well on anti-seizure medication and are able to resume a normal lifestyle. Some patients continue to experience periodic break-through seizures. Many dogs require occasional medication adjustments, and some require the addition of other medications over time.")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/epilepsy-in-dogs")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hypothyroidism")
                    st.image("https://www.lifelearn-cliented.com//cms/resources/body/24023/2024_817i_thyroid_dog_5002.png")
                with col2:
                    Hypothyroidism = ''' A condition of inadequate thyroid hormone levels that leads to a reduction in a dog's metabolic state. Hypothyroidism is one of the most common hormonal (endocrine) diseases in dogs. It generally affects middle-aged dogs (average of 6–7 years of age), and it may be more common in spayed females and neutered males. A wide variety of breeds may be affected.'''
                    st.markdown(Hypothyroidism)
                with st.expander("See More Details"):
                    st.subheader("What causes hypothyroidism?")
                    st.write("In dogs, hypothyroidism is usually caused by one of two diseases: lymphocytic thyroiditis or idiopathic thyroid gland atrophy. **Lymphocytic thyroiditis** is the most common cause of hypothyroidism and is thought to be an immune-mediated disease, meaning that the immune system decides that the thyroid is abnormal or foreign and attacks it. It is unclear why this occurs; however, it is a heritable trait, so genetics plays a role. In **idiopathic thyroid gland atrophy**, normal thyroid tissue is replaced by fat tissue. This condition is also poorly understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of hypothyroidism?")
                    st.write("When the metabolic rate slows down, virtually every organ in the body is affected. Most dogs with hypothyroidism have one or more of the following signs:")
                    st.write("weight gain without an increase in appetite")
                    st.write("lethargy (tiredness) and lack of desire to exercise")
                    st.write("cold intolerance (gets cold easily)")
                    st.write("dry, dull hair with excessive shedding")
                    st.write("very thin to nearly bald hair coat")
                    st.write("increased dark pigmentation in the skin")
                    st.write("increased susceptibility and occurrence of skin and ear infections")
                    st.write("failure to re-grow hair after clipping or shaving")
                    st.write("high blood cholesterol")
                    st.write("slow heart rate")
                    st.markdown("---")
                    st.subheader("How is hypothyroidism diagnosed?")
                    st.write("The most common screening test is a total thyroxin (TT4) level. This is a measurement of the main thyroid hormone in a blood sample. A low level of TT4, along with the presence of clinical signs, is suggestive of hypothyroidism. Definitive diagnosis is made by performing a free T4 by equilibrium dialysis (free T4 by ED) or a thyroid panel that assesses the levels of multiple forms of thyroxin. If this test is low, then your dog has hypothyroidism. Some pets will have a low TT4 and normal free T4 by ED. These dogs do not have hypothyroidism. Additional tests may be necessary based on your pet's condition. See handout “Thyroid Hormone Testing in Dogs” for more information.")
                    st.markdown("---")
                    st.subheader("Can it be treated?")
                    st.write("Hypothyroidism is treatable but not curable. It is treated with oral administration of thyroid replacement hormone. This drug must be given for the rest of the dog's life. The most recommended treatment is oral synthetic thyroid hormone replacement called levothyroxine (brand names Thyro-Tabs® Canine, Synthroid®).")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hypothyroidism-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Syncope")
                    st.image("https://image.petmd.com/files/styles/978x550/public/2023-03/iStock-515603112.jpg?w=1080&q=75")
                with col2:
                    Syncope = ''' Tthe medical term for fainting, which can occur both in dogs (and cats), due to a lack of oxygen or nutrients to the brain. Syncope in dogs is considered a medical emergency—immediate treatment is urgent and critical.'''
                    st.markdown(Syncope)
                with st.expander("See More Details"):
                    st.subheader("Symptoms of Syncope in Dogs")
                    st.write(" Syncope is considered to be more of a clinical sign than a disease, and occurs when there is a sudden loss of consciousness associated with the collapse of the dog. It is temporary, and dogs usually recover after only a few seconds to minutes.")
                    st.markdown("---")
                    st.subheader("Causes of Syncope in Dogs")
                    st.write("Multiple conditions and underlying diseases can cause fainting in dogs:")
                    st.write("**Cardiovascular conditions** including arrhythmias, heart failure, dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), mitral valve disease, pericardial effusion, pulmonary hypertension, and congenital heart defects. Other conditions such as cancer or diseases that affect blood output (e.g., heartworm disease) can have similar effects.")
                    st.write("**Neurologic conditions** including brain tumors, vascular disease, and narcolepsy")
                    st.write("**Acute hemorrhage** (blood loss) or profound anemia (low red blood cell count)")
                    st.write("**Hypoglycemia** (low blood sugar)­ or electrolyte abnormalities")
                    st.write("Adverse or known drug side effects, from drugs such as vasodilators or beta blockers")
                    st.write("**Situational syncope**, often due to a sudden change in pressure within the body from an event such as coughing, vomiting, urination, or defecation. Pulling tightly on a dog’s collar or leash can have similar effects, though due to a different mechanism.")
                    st.write("**Vasovagal syncope**, due to a sudden drop in blood pressure often preceded by some sort of stressful or emotional situation that causes a change in body reflexes")
                    st.markdown("---")
                    st.subheader("How Veterinarians Diagnose Syncope in Dogs")
                    st.write("Your dog’s clinical history and physical exam can often not only facilitate a diagnosis of syncope but also provide insight into an underlying cause as well as a way forward for possible treatment options. It’s always helpful if you can provide a detailed account of the event itself, as well as what occurred before and after the event.")
                    st.write("Bloodwork and urine testing will most likely be recommended, as well as a heart work-up including chest X-rays, EKG (a 24-hour Holter monitor may be needed for the diagnosis of arrhythmias), blood pressure, and echocardiogram. Referral to a veterinary cardiologist or neurologist to consider neurologic conditions with a test such as CT, MRI, or CSF tap may also be discussed and recommended.")
                    st.markdown("---")
                    st.subheader("Treatment for Syncope in Dogs")
                    st.write("The prognosis is variable and often related to the underlying cause. For dogs suffering from situational syncope, limiting or preventing provoking situations is key. For instance, if syncopal events occur every time your dog becomes excited when the doorbell rings, then disconnecting the doorbell or installing another system may be the appropriate fix. Switching to a harness instead of a collar or leash is also advisable.")
                    st.write("Treatment for dogs suffering from a heart condition will also vary. Arrhythmias or heart failure will often benefit from medications such as sotalol, an anti-arrhythmic, or enalapril, which helps regulate blood pressure. Surgical procedures may be warranted for dogs requiring a pacemaker or in cases of a blockage or tumor, or when needed to resolve pericardial effusion. Chemotherapy or radiation may be necessary to treat cancer.")
                    st.write("Dogs showing electrolyte abnormalities or low blood sugar may require IV fluids and supplementation. Blood loss will require a transfusion. Discontinuing medications that could be causing a condition may also be recommended.")
                    st.markdown("---")
                    st.link_button("Source","https://www.petmd.com/dog/conditions/neurological/syncope-fainting-dogs")

        elif breed_label == "Norwich Terrier":
            tab1, tab2, tab3= st.tabs(["Hypothyroidism", "Lens luxation", "Corneal dystrophy"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hypothyroidism")
                    st.image("https://www.lifelearn-cliented.com//cms/resources/body/24023/2024_817i_thyroid_dog_5002.png")
                with col2:
                    Hypothyroidism = ''' A condition of inadequate thyroid hormone levels that leads to a reduction in a dog's metabolic state. Hypothyroidism is one of the most common hormonal (endocrine) diseases in dogs. It generally affects middle-aged dogs (average of 6–7 years of age), and it may be more common in spayed females and neutered males. A wide variety of breeds may be affected.'''
                    st.markdown(Hypothyroidism)
                with st.expander("See More Details"):
                    st.subheader("What causes hypothyroidism?")
                    st.write("In dogs, hypothyroidism is usually caused by one of two diseases: lymphocytic thyroiditis or idiopathic thyroid gland atrophy. **Lymphocytic thyroiditis** is the most common cause of hypothyroidism and is thought to be an immune-mediated disease, meaning that the immune system decides that the thyroid is abnormal or foreign and attacks it. It is unclear why this occurs; however, it is a heritable trait, so genetics plays a role. In **idiopathic thyroid gland atrophy**, normal thyroid tissue is replaced by fat tissue. This condition is also poorly understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of hypothyroidism?")
                    st.write("When the metabolic rate slows down, virtually every organ in the body is affected. Most dogs with hypothyroidism have one or more of the following signs:")
                    st.write("weight gain without an increase in appetite")
                    st.write("lethargy (tiredness) and lack of desire to exercise")
                    st.write("cold intolerance (gets cold easily)")
                    st.write("dry, dull hair with excessive shedding")
                    st.write("very thin to nearly bald hair coat")
                    st.write("increased dark pigmentation in the skin")
                    st.write("increased susceptibility and occurrence of skin and ear infections")
                    st.write("failure to re-grow hair after clipping or shaving")
                    st.write("high blood cholesterol")
                    st.write("slow heart rate")
                    st.markdown("---")
                    st.subheader("How is hypothyroidism diagnosed?")
                    st.write("The most common screening test is a total thyroxin (TT4) level. This is a measurement of the main thyroid hormone in a blood sample. A low level of TT4, along with the presence of clinical signs, is suggestive of hypothyroidism. Definitive diagnosis is made by performing a free T4 by equilibrium dialysis (free T4 by ED) or a thyroid panel that assesses the levels of multiple forms of thyroxin. If this test is low, then your dog has hypothyroidism. Some pets will have a low TT4 and normal free T4 by ED. These dogs do not have hypothyroidism. Additional tests may be necessary based on your pet's condition. See handout “Thyroid Hormone Testing in Dogs” for more information.")
                    st.markdown("---")
                    st.subheader("Can it be treated?")
                    st.write("Hypothyroidism is treatable but not curable. It is treated with oral administration of thyroid replacement hormone. This drug must be given for the rest of the dog's life. The most recommended treatment is oral synthetic thyroid hormone replacement called levothyroxine (brand names Thyro-Tabs® Canine, Synthroid®).")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hypothyroidism-in-dogs")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:  
                    st.header("Lens luxation")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Lens_luxation = '''A painful and potentially blinding inherited canine eye condition. Lens luxation occurs when the ligaments supporting the lens weaken, displacing it from its normal position. Signs of lens luxation may include red, teary, hazy, or cloudy, painful eyes. PLL can cause eye inflammation and glaucoma, particularly if the lens shifts forward into the eye. '''
                    st.markdown(Lens_luxation)
                with st.expander("See More Details"):
                    st.subheader("Causes")
                    st.write("The lens is a structure in the eye located behind the iris (the colored portion of the eye) responsible for focusing light onto the retina for visualization. It is suspended in the eye by multiple ligaments called zonules. PLL is caused by an inherited weakness and breakdown of the zonules, displacing the lens from its normal position in the eye. The direction that the lens luxates can be either forward (anterior) or backward (posterior). Anterior lens luxation is the most damaging and considered an emergency as it can rapidly increase pressure inside the eye, known as glaucoma, causing pain and potentially blindness. Posterior lens luxation leads to milder inflammation, and glaucoma is less likely to develop.")
                    st.write("PLL most commonly develops in dogs between the ages of three and eight. However, structural changes in the eye may already be evident at 20 months of age, long before lens luxation typically occurs. Both eyes are often affected by PLL, but not necessarily at the same time. This differs from secondary lens luxation, which can more commonly only affect one eye and is usually caused by a coexisting ocular disease such as glaucoma, inflammatory conditions of the eye (uveitis), cataracts, eye trauma and eye tumors.")
                    st.markdown("---")
                    st.subheader("Diagnosis")
                    st.write("Early detection of lens luxation is crucial. Your veterinarian will diagnose primary lens luxation by performing a complete eye exam. They may measure your dog’s eye pressure for secondary conditions like glaucoma. You may be referred to a veterinary ophthalmology specialist where additional testing could include an eye ultrasound to evaluate the internal structures of the eye.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("Treatment options vary by stage of disease and position of the lens. When diagnosed early, the most common treatment for anterior lens luxation is surgery to remove the lens by a veterinarian specializing in ophthalmology. Topical eye medications may be needed long-term, even after surgery.")
                    st.write("If glaucoma develops suddenly, this requires emergency management and may include medication to decrease eye pressure, followed by referral to a veterinary ophthalmologist. If the eye has uncontrolled glaucoma, is permanently blind, or there is pain or inflammation, it may be necessary for the affected eye to be surgically removed (enucleation).")
                    st.write("Treatment for posterior lens luxation may include topical medications to help prevent the lens from shifting forward and causing more severe damage to the eye.")
                    st.markdown("---")
                    st.subheader("Outcome")
                    st.write("Primary lens luxation most commonly progresses to affect both eyes. For this reason, regular and in-depth ocular examinations are recommended in at-risk dogs. Anterior lens luxation left untreated or not addressed immediately often has a poor prognosis for saving the eye.")
                    st.write("Dogs that receive surgery early for anterior lens luxation can often preserve some vision but may have diminished vision that is more blurred up close. However, this doesn’t generally appear to affect everyday life. Surgery is not without risk of complications, and often, patients require lifelong topical eye medications.")
                    st.markdown("---")
                    st.link_button("Source","https://www.vet.cornell.edu/departments/riney-canine-health-center/canine-health-information/primary-lens-luxation#:~:text=Lens%20luxation%20occurs%20when%20the,shifts%20forward%20into%20the%20eye.")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:  
                    st.header("Corneal dystrophy")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Corneal_dystrophy = '''Caused by a genetic disturbance in how fat is metabolized. The result is a white or gray clouding of the eye. It generally starts in one eye but always affects both. In most breeds, it does not cause discomfort or blindness.'''
                    st.markdown(Corneal_dystrophy)
                with st.expander("See More Details"):
                    st.subheader("Symptoms of Corneal Dystrophy in Dogs")
                    st.write("At the onset, corneal dystrophy usually appears as a white or grayish round “cloud” at the center of the eye. If ulcers are present the dog may give signs of irritated eyes, rubbing, and itching. The spots are usually round but sometimes donut shaped. Symptoms vary widely between breeds.  The affliction can seemingly appear at any age,in as little as four months in Airedale Terriers and as late as thirteen years in Chihuahuas. In some breeds, the trait is thought to be sex linked.")
                    st.markdown("---")
                    st.subheader("Causes of Corneal Dystrophy in Dogs")
                    st.write("Corneal dystrophy is an inherited condition affecting the ability of the cells to process fat. It is an autosomal recessive trait, meaning both the dam and the sire must carry the gene in order for the puppy to be affected, at least in some breeds. In other breeds, the mode of inheritance appears to be sex-linked. In still other breeds, the mode of inheritance has not been identified.")
                    st.markdown("---")
                    st.subheader("Diagnosis of Corneal Dystrophy in Dogs")
                    st.write("Diagnosis is made from the observation of the lesion. This can be done by the use of a fluorescein dye which may clearly define the problem.  Further testing of the eye may include intraocular pressure and tear test. Blood work is often done to verify markers in the blood consistent with this condition, such as cholesterol. An eye specialist may be brought on board by your veterinarian to rule out other corneal diseases or degradation")
                    st.markdown("---")
                    st.subheader("Treatment of Corneal Dystrophy in Dogs")
                    st.write("In most cases, treatment is not needed. If the condition does not progress rapidly, cause the dog discomfort, or affect vision, often the best course is to leave the eye alone. Your dog may notice the spot on his eye for a while but his brain will train him to see past it without annoyance much like your brain will do the same for you. ")
                    st.write("Because the condition has to do with the process of fat, sometimes a low fat, high fiber diet is recommended. There is some disagreement among researchers as to whether or not a low-fat diet is effective. The general consensus is that the fat should be lower than 10% in dry matter (kibble) and adherence in all foods and treats is needed to see results. ")
                    st.write("Sometimes in cases of corneal dystrophy a topical acid treatment (TCA) may be recommended. This treatment may be done once or more times to aid in comfort. It helps to dissolve the mineral deposits that leads to ulcers.")
                    st.write("In severe cases of corneal dystrophy, surgery to remove the mineral deposits can be recommended. As with any surgery, complications can arise. At times, scar tissue remains where the mineral deposits were. Other more severe complications can lead to rupture of the eye or retinal detachment. Although complications are rare, some can lead to blindness. Because corneal dystrophy sometimes is associated with Cushing’s disease, testing for that should be done after diagnosis.")
                    st.markdown("---")
                    st.link_button("Source","https://wagwalking.com/condition/corneal-dystrophy")

        elif breed_label == "Yorkshire terrier":  
            tab1, tab2, tab3= st.tabs(["Cataract", "Cryptorchidism", "Demodicosis"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cryptorchidism")
                    st.image("https://iloveveterinary.com/wp-content/uploads/2021/05/Cryptorchid-Chihuahua.jpg.webp", width=200)
                with col2:
                    Cryptorchidism = '''the medical term that refers to the failure of one or both testicles (testes) to descend into the scrotum. The testes develop near the kidneys within the abdomen and normally descend into the scrotum by two months of age. In certain dogs, it may occur later, but rarely after six months of age. Cryptorchidism may be presumed to be present if the testicles cannot be felt in the scrotum after two to four months of age.
                            '''
                    st.markdown(Cryptorchidism)
                with st.expander("See More details"):
                    st.subheader("If the testicles aren't in the scrotum, where are they?")
                    st.write("In most cases of cryptorchidism, the testicle is retained in the abdomen or in the inguinal canal (the passage through the abdominal wall into the genital region through which a testicle normally descends). Sometimes, the testicle will be located in the subcutaneous tissues (just under the skin) in the groin region, between the inguinal canal and the scrotum.")
                    st.markdown("---")
                    st.subheader("How is cryptorchidism diagnosed?")
                    st.write("In cases of abdominal cryptorchidism, the testicle cannot be felt from the outside. An abdominal ultrasound or radiographs (X-rays) may be performed to determine the exact location of the retained testicle, but this is not often done before surgery, as it is not required to proceed with surgery. Typically, only one testicle is retained, and this is called unilateral cryptorchidism. If you have a dog that does not appear to have testicles but is exhibiting male behaviors, a hormonal test called an hCG stimulation test can be performed to see if he is already neutered.")
                    st.markdown("---")
                    st.subheader("What causes cryptorchidism and how common is it?")
                    st.write("Cryptorchidism occurs in all breeds but toy breeds, including toy Poodles, Pomeranians, and Yorkshire Terriers, may be at higher risk. Approximately 75% of cases of cryptorchidism involve only one retained testicle while the remaining 25% involve failure of both testicles to descend into the scrotum. The right testicle is more than twice as likely to be retained as the left testicle. Cryptorchidism affects approximately 1-3% of all dogs. The condition appears to be inherited since it is commonly seen in families of dogs, although the exact cause is not fully understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of cryptorchidism?")
                    st.write("This condition is rarely associated with pain or other signs unless a complication develops. In its early stages, a single retained testicle is significantly smaller than the other, normal testicle. If both testicles are retained, the dog may be infertile. The retained testicles continue to produce testosterone but generally fail to produce sperm.")
                    st.markdown("---")
                    st.subheader("What is the treatment for cryptorchidism?")
                    st.write("Neutering and removal of the retained testicle(s) are recommended. If only one testicle is retained, the dog will have two incisions - one for extraction of each testicle. If both testicles are in the inguinal canal, there will also be two incisions. If both testicles are in the abdomen, a single abdominal incision will allow access to both.")
                    st.markdown("---")
                    st.subheader("What if I don't want to neuter my dog?")
                    st.write("There are several good reasons for neutering a dog with cryptorchidism. The first reason is to remove the genetic defect from the breed line. Cryptorchid dogs should never be bred. Second, dogs with a retained testicle are more likely to develop a testicular tumor (cancer) in the retained testicle. Third, as described above, the testicle can twist, causing pain and requiring emergency surgery to correct. Finally, dogs with a retained testicle typically develop the undesirable characteristics associated with intact males like urine marking and aggression. The risk of developing testicular cancer is estimated to be at least ten times greater in dogs with cryptorchidism than in normal dogs.")
                    st.markdown("---")
                    st.subheader("What is the prognosis for a dog with cryptorchidism?")
                    st.write("The prognosis is excellent for dogs that undergo surgery early before problems develop in the retained testicle. The surgery is relatively routine, and the outcomes are overwhelmingly positive.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/retained-testicle-cryptorchidism-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Demodicosis")
                    st.image("https://images.wagwalkingweb.com/media/articles/dog/demodex-/demodex-.jpg?auto=compress&fit=max&width=640")
                with col2:
                    Demodicosis = '''are cigar shaped microscopic parasitic mites that live within the hair follicles of all dogs. These mites are passed to puppies from their mothers in the first few days of life, and then live within the hair follicles for the duration of animal’s life without causing problems'''
                    st.markdown(Demodicosis)
                with st.expander("See More Details"):
                    st.subheader("What causes demodicosis?")
                    st.write("There are two presentations of demodicosis depending on the age at which it develops. Juvenile onset demodicosis tends to occur in puppyhood between the ages of 3 months and 18 months, and occurs in both localised and generalised forms. The exact cause is quite poorly understood but probably occurs due to a mite specific genetic defect in the immune system which allows mite numbers to increase. This defect may or may not resolve as the puppy ages. It is thought to be ‘mite specific’ because these puppies are healthy in all other respects and do not succumb to other infections. Generalised demodicosis can be a very severe disease. Adult onset demodicosis usually occurs in the generalised form and in dogs over 4 years of age. It is generally considered a more severe disease than its juvenile onset counterpart. In these cases, mite numbers have been controlled in normal numbers in the hair follicles for years prior to the onset of disease, which tends to result from a systemic illness affecting the immune system. Common triggers for adult onset demodicosis include hormonal diseases and cancer.")
                    st.markdown("---")
                    st.subheader("What are the clinical signs?")
                    st.write("Localised demodicosis in juvenile dogs presents as patches of hair loss and red inflamed skin. These patches often occur around the face, head and feet and are not typically itchy")
                    st.markdown("---")
                    st.subheader("How is it diagnosed?")
                    st.write("Demodicosis can often be suspected following a review of the animal’s history and assessment of the clinical signs. The parasitic mites within the hair follicles result in plugging and the formation of ‘black heads’. The plugged follicles also cause large amounts of scale to be present on the hairs themselves.")
                    st.write("Demodicosis can usually be diagnosed relatively easily. Hairs can be plucked from the affected skin and then examined under a microscope for the presence of the mites. Alternatively, the skin can be squeezed and then scraped with a blade to collect up the surface debris from the skin. This material is then also examined under a microscope for the parasites.")
                    st.markdown("---")
                    st.subheader("Is it contagious?")
                    st.write("Demodex mites from dogs are considered non-infectious to in-contact animals and people. It is thought that Demodex mites can only be passed between dogs in the first few days of life from the mother to the pup.")
                    st.markdown("---")
                    st.subheader("How is it treated?")
                    st.write("The treatment used for demodicosis depends on the age of the animal and the severity of the disease. Mild and localised forms of demodicosis in young dogs may not require treatment, and may resolve spontaneously as the animal ages. These cases should be closely monitored if no treatment is given.")
                    st.write("Generalised cases in young dogs and those in adult dogs require intensive treatment. Secondary infections must be treated with courses of antibiotics, and a swab is often submitted to a laboratory to grow the organisms to ensure the correct antibiotic is selected. The licensed treatments for demodicosis in the UK include a dip solution called Aludex and a spot-on product called Advocate. The dip is performed on a weekly basis until mite numbers are brought under control. Advocate spot-on is generally used for milder cases and is usually used monthly. In severe cases not responding to the licensed treatments, off-licence treatments must be used. Some of these drugs, such as Ivermectin and Milbemycin, are used for demodicosis in other countries.")
                    st.markdown("---")
                    st.subheader("What is the prognosis?")
                    st.write("The prognosis for localised disease in young dogs is very good, and most recover uneventfully from the disease. Generalised cases in young dogs can take many weeks or even months of treatment, but it is usually possible to control the disease with a good long term outlook.")
                    st.write("The prognosis for adult onset generalised demodicosis is far more uncertain, as many of these dogs have an underlying systemic illness. If this illness can be identified and cured, the prognosis for managing the demodicosis is much better. Some cases require long term medication to keep mite numbers controlled.")
                    st.markdown("---")
                    st.link_button("Source","https://www.ndsr.co.uk/information-sheets/canine-demodicosis/#:~:text=And%20what%20is%20demodicosis%3F,animal's%20life%20without%20causing%20problems.")

        elif breed_label == "Wire haired fox terrier":
            tab1, tab2, tab3= st.tabs(["Atopic dermatitis", "Cataract", "Entropion"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Atopic dermatitis")
                    st.image("https://images.ctfassets.net/82d3r48zq721/6jiyt1463nRXCZa451La8T/5114391b07c7760c8b9068224869ae42/Dog-with-skin-allergy-atopic-dematitis-biting-and-scratching-himself_resized.jpg?w=800&h=531&q=50&fm=webp")
                with col2:
                    Atopic_dermatitis = '''Atopic dermatitis (or atopy) is an inflammatory, chronic skin condition associated with environmental allergies and is the second most common allergic skin condition diagnosed in dogs.'''
                    st.markdown(Atopic_dermatitis)
                with st.expander("See More Details"):
                    st.subheader("Signs & symptoms of atopic dermatitis in dogs")
                    st.write("A dog with atopic dermatitis will usually show signs and symptoms between 3 months to 6 years of age. It’s not as common for dogs over the age of 7 to develop atopic dermatitis, although a new environment can trigger new allergens. Atopic dermatitis often begins as a mild condition with symptoms not becoming clinically visible before the third year.")
                    st.markdown("---")
                    st.subheader("What causes atopic dermatitis in dogs?")
                    st.write("Atopic dermatitis is a genetic disease that is predisposed in some breeds more than others. For that reason, dogs diagnosed with the condition should not be bred. The cause is unknown, but a general understanding of the anatomy of the skin is vital in understanding what happens to a dog when the skin becomes irritated and inflamed as a result of allergens in the environment. A case of atopic dermatitis can be painful and uncomfortable for a dog.")
                    st.markdown("---")
                    st.subheader("Diagnosing canine atopic dermatitis")
                    st.write("Symptoms of atopic dermatitis are similar to other skin conditions, which can make it difficult to diagnose. Uncovering the cause may take time and is often a process of elimination. Along with a full medical examination, which includes a look at the dog’s complete medical history, additional allergy testing may be done. In some cases, your veterinarian may perform a blood test (serological allergy test) to determine the presence of an antibody called IgE to specific allergens. An increase in an allergen-specific IgE usually means there is an overreaction to that allergen in the body.")
                    st.markdown("---")
                    st.subheader("Treatment for atopic dermatitis in dogs")
                    st.write("One of the first steps is eliminating or reducing exposure to the allergens causing dermatitis. If you are unable to identify the irritants, use a process of elimination by removing the environmental factors that have the potential to trigger an outbreak. Diet, bedding, even the general environment in which the dog is exposed to may need to be changed.")
                    st.write("For dogs with a severe case of atopic dermatitis, removing and changing specific factors might not be enough. Oral corticosteroids can be given to control or reduce the itching and swelling, but there are side effects associated with steroids, so it’s important to administer as directed by your veterinarian. There are also other non-steroidal drugs that your veterinarian might prescribe to alleviate the discomfort.")
                    st.markdown("---")
                    st.link_button("Source","https://www.smalldoorvet.com/learning-center/medical/atopic-dermatitis-in-dogs")
            with tab2:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Entropion")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Entroption = '''when the eyelids roll inward toward the eye. The fur on the eyelids and the eyelashes then rub against the surface of the eye (the cornea). This is a very painful condition that can lead to corneal ulcers.'''
                    st.markdown(Entroption)
                with st.expander("See More Details"):
                    st.write("Many Bloodhounds have abnormally large eyelids (macroblepharon) which results in an unusually large space between the eyelids.  Because of their excessive facial skin and resulting facial droop, there is commonly poor support of the outer corner of the eyelids")
                    st.markdown("---")
                    st.subheader("How is entropion treated?")
                    st.write("The treatment for entropion is surgical correction. A section of skin is removed from the affected eyelid to reverse its inward rolling. In many cases, a primary, major surgical correction will be performed, and will be followed by a second, minor corrective surgery later. Two surgeries are often performed to reduce the risk of over-correcting the entropion, resulting in an outward-rolling eyelid known as ectropion. Most dogs will not undergo surgery until they have reached their adult size at six to twelve months of age.")
                    st.markdown("---")
                    st.subheader("Should an affected dog be bred?")
                    st.write("Due to the concern of this condition being inherited, dogs with severe ectropion requiring surgical correction should not be bred.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/eyelid-entropion-in-dogs")

        elif breed_label == "Lakeland Terrier":
            tab1, tab2, tab3= st.tabs(["Lens luxation", "Undershot jaw", "Ununited anconeal process"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:  
                    st.header("Lens luxation")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Lens_luxation = '''A painful and potentially blinding inherited canine eye condition. Lens luxation occurs when the ligaments supporting the lens weaken, displacing it from its normal position. Signs of lens luxation may include red, teary, hazy, or cloudy, painful eyes. PLL can cause eye inflammation and glaucoma, particularly if the lens shifts forward into the eye. '''
                    st.markdown(Lens_luxation)
                with st.expander("See More Details"):
                    st.subheader("Causes")
                    st.write("The lens is a structure in the eye located behind the iris (the colored portion of the eye) responsible for focusing light onto the retina for visualization. It is suspended in the eye by multiple ligaments called zonules. PLL is caused by an inherited weakness and breakdown of the zonules, displacing the lens from its normal position in the eye. The direction that the lens luxates can be either forward (anterior) or backward (posterior). Anterior lens luxation is the most damaging and considered an emergency as it can rapidly increase pressure inside the eye, known as glaucoma, causing pain and potentially blindness. Posterior lens luxation leads to milder inflammation, and glaucoma is less likely to develop.")
                    st.write("PLL most commonly develops in dogs between the ages of three and eight. However, structural changes in the eye may already be evident at 20 months of age, long before lens luxation typically occurs. Both eyes are often affected by PLL, but not necessarily at the same time. This differs from secondary lens luxation, which can more commonly only affect one eye and is usually caused by a coexisting ocular disease such as glaucoma, inflammatory conditions of the eye (uveitis), cataracts, eye trauma and eye tumors.")
                    st.markdown("---")
                    st.subheader("Diagnosis")
                    st.write("Early detection of lens luxation is crucial. Your veterinarian will diagnose primary lens luxation by performing a complete eye exam. They may measure your dog’s eye pressure for secondary conditions like glaucoma. You may be referred to a veterinary ophthalmology specialist where additional testing could include an eye ultrasound to evaluate the internal structures of the eye.")
                    st.markdown("---")
                    st.subheader("Treatment")
                    st.write("Treatment options vary by stage of disease and position of the lens. When diagnosed early, the most common treatment for anterior lens luxation is surgery to remove the lens by a veterinarian specializing in ophthalmology. Topical eye medications may be needed long-term, even after surgery.")
                    st.write("If glaucoma develops suddenly, this requires emergency management and may include medication to decrease eye pressure, followed by referral to a veterinary ophthalmologist. If the eye has uncontrolled glaucoma, is permanently blind, or there is pain or inflammation, it may be necessary for the affected eye to be surgically removed (enucleation).")
                    st.write("Treatment for posterior lens luxation may include topical medications to help prevent the lens from shifting forward and causing more severe damage to the eye.")
                    st.markdown("---")
                    st.subheader("Outcome")
                    st.write("Primary lens luxation most commonly progresses to affect both eyes. For this reason, regular and in-depth ocular examinations are recommended in at-risk dogs. Anterior lens luxation left untreated or not addressed immediately often has a poor prognosis for saving the eye.")
                    st.write("Dogs that receive surgery early for anterior lens luxation can often preserve some vision but may have diminished vision that is more blurred up close. However, this doesn’t generally appear to affect everyday life. Surgery is not without risk of complications, and often, patients require lifelong topical eye medications.")
                    st.markdown("---")
                    st.link_button("Source","https://www.vet.cornell.edu/departments/riney-canine-health-center/canine-health-information/primary-lens-luxation#:~:text=Lens%20luxation%20occurs%20when%20the,shifts%20forward%20into%20the%20eye.")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:  
                    st.header("Undershot jaw")
                    st.image("https://vhd-wordpress-uploads.s3.amazonaws.com/uploads/2023/08/canine-teeth-232551_1280.jpg")
                with col2:
                    Undershot_jaw = '''A condition where the lower jaw is too long for the upper jaw.'''
                    st.markdown(Undershot_jaw)
                with st.expander("See More Details"):
                    st.subheader("Symptom")
                    st.write("While this condition is normal for some breeds, such as Bulldogs, in many breeds it is unusual. An undershot jaw occurs when the lower jaw grows faster than normal and becomes longer than the upper jaw, and is usually evident around 8 weeks of age in puppies.")
                    st.markdown("---")
                    st.subheader("Cause")
                    st.write("The cause of overshot and undershot jaws in dogs relate to the increased or decreased rate of growth of the upper and lower jaws in relation to one another. This can occur due to a: Genetic disorder Trauma; Systemic infection ;Nutritional disorder; Endocrine disorder; Abnormal setting of puppy teeth; Early or late loss of puppy teeth.")
                    st.markdown("---")
                    st.subheader("Diagnosing")
                    st.write("After a quick physical exam, your vet may have to sedate your dog in order to perform a thorough oral exam. This will assess your dog’s skull type and teeth location in relation to the teeth on the opposite jaw. Often, the placement of the upper and lower incisors in relation to one another can determine what type of malocclusion your dog has. Your vet will note any areas of trauma due to teeth striking those areas, and any cysts, tumors, abscesses, or remaining puppy teeth that may be present. A dental X-ray can also help to assess the health of the jaws and teeth. These diagnostic methods will lead to a diagnosis of an overshot or undershot jaw in your dog.")
                    st.markdown("---")
                    st.subheader("Treat Method")
                    st.write("Treatment of a jaw misalignment will depend on the severity of the condition. If your dog has a misalignment, but can still bite and chew food without problems, no treatment may be needed. If the misalignment is caught early in a puppy’s life, it may only be temporary and may correct itself over time. However, there are times when intervention may be needed. If your puppy’s teeth are stopping the normal growth of his jaws, then surgery to remove those puppy teeth may be performed. This may allow the jaws to continue to grow, but will not make them grow. For older dogs who are experiencing pain and trauma due to misaligned jaws and teeth, oral surgery is generally performed to extract teeth that are causing trauma, to move teeth so that they fit, or to create space for a misaligned tooth to occupy. Other therapies include crown reductions or braces.")
                    st.link_button("Source","https://ngdc.cncb.ac.cn/idog/disease/getDiseaseDetailById.action?diseaseId=319#:~:text=While%20this%20condition%20is%20normal,weeks%20of%20age%20in%20puppies.")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:  
                    st.header("Ununited anconeal process")
                    st.image("https://vhd-wordpress-uploads.s3.amazonaws.com/uploads/2023/08/canine-teeth-232551_1280.jpg")
                with col2:
                    Ununited_anconeal_process = '''Ununited anconeal process is a condition in which a bony protuberance within the elbow becomes detached from the ulna. This loose, bony fragment causes pain and lameness and contributes to articular damage within the elbow joint.'''
                    st.markdown(Ununited_anconeal_process)
                with st.expander("See More Details"):
                    st.subheader("What causes UAP?")
                    st.write("This condition is most commonly diagnosed in German Shepherd Dogs and Bernese Mountain Dogs. It is a consequence of elbow incongruity (such as ulnar notch incongruity or short ulna syndrome). In most dogs, this causes obvious lameness and pain. Achieving an optimal outcome depends on early diagnosis and treatment.")
                    st.markdown("---")
                    st.subheader("How is UAP diagnosed?")
                    st.write("Ununited anconeal process is most commonly diagnosed using simple radiography. However, in some subtle cases where the anconeal process remains loosely attached, it is necessary to use computed topography (CT) to diagnose the condition.")
                    st.markdown("---")
                    st.subheader("How is UAP treated?")
                    st.write("n dogs affected by ununited anconeal process (UAP), the best chance of achieving an excellent outcome is by early surgical intervention. Surgery can allow reattachment of the anconeal process but is only appropriate in dogs in which diagnosis is made early and where the loose anconeal process has not changed shape. In chronic cases of UAP where the fragment has changed shape, the most appropriate treatment is usually anconeal process removal. Fitzpatrick Referrals has employed a unique type of screw for anconeal process reattachment. This technique is combined with a proximal ulnar osteotomy (PUO) to allow the proximal ulna to move to a more favourable position relative to the humerus. In some select patients, PUO alone without a screw may be sufficient to allow the anconeal process to fuse to the ulna.")
                    st.markdown("---")
                    st.link_button("Source","https://www.fitzpatrickreferrals.co.uk/orthopaedic/ununited-anconeal-process-uap/#:~:text=Ununited%20anconeal%20process%20is%20a,damage%20within%20the%20elbow%20joint.")

        elif breed_label == "Sealyham Terrier": 
            tab1, tab2, tab3= st.tabs(["Cataract", "Glaucoma", "Hypothyroidism"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Glaucoma")
                    st.image("https://www.animaleyecare.com.au/images/animal-eye-care/conditions/glaucoma-in-dogs-w.jpg")
                with col2:
                    Glaucoma = '''A disease of the eye in which the pressure within the eye, called intraocular pressure (IOP), is increased. Intraocular pressure is measured using an instrument called a tonometer.'''
                    st.markdown(Glaucoma)
                with st.expander("See More Details"):
                    st.subheader("What is intraocular pressure and how is it maintained?")
                    st.write("The inside of the eyeball is filled with fluid, called aqueous humor. The size and shape of the normal eye is maintained by the amount of fluid contained within the eyeball. The pressure of the fluid inside the front or anterior chamber of the eye is known as intraocular pressure (IOP). Aqueous humor is produced by a structure called the ciliary body. In addition to producing this fluid (aqueous humor), the ciliary body contains the suspensory ligaments that hold the lens in place. Muscles in the ciliary body pull on the suspensory ligaments, controlling the shape and focusing ability of the lens.Aqueous humor contains nutrients and oxygen that are used by the structures within the eye. The ciliary body constantly produces aqueous humor, and the excess fluid is constantly drained from the eye between the cornea and the iris. This area is called the iridocorneal angle, the filtration angle, or the drainage angle.As long as the production and absorption or drainage of aqueous humor is equal, the intraocular pressure remains constant.")
                    st.markdown('---') 
                    st.subheader("What causes glaucoma?")
                    st.write("Glaucoma is caused by inadequate drainage of aqueous fluid; it is not caused by overproduction of fluid. Glaucoma is further classified as primary or secondary glaucoma.")
                    st.write(f"**Primary glaucoma** results in increased intraocular pressure in a healthy eye. Some breeds are more prone than others (see below). It occurs due to inherited anatomical abnormalities in the drainage angle.")
                    st.write(f"**Secondary glaucoma** results in increased intraocular pressure due to disease or injury to the eye. This is the most common cause of glaucoma in dogs. Causes include:")
                    st.write(f"**Uveitis** (inflammation of the interior of the eye) or severe intraocular infections, resulting in debris and scar tissue blocking the drainage angle.")
                    st.write(f"**Anterior dislocation of lens**. The lens falls forward and physically blocks the drainage angle or pupil so that fluid is trapped behind the dislocated lens.")
                    st.write(f"**Tumors** can cause physical blockage of the iridocorneal angle.")
                    st.write(f"**Intraocular bleeding.** If there is bleeding in the eye, a blood clot can prevent drainage of the aqueous humor.")
                    st.write(f"Damage to the lens. Lens proteins leaking into the eye because of a ruptured lens can cause an inflammatory reaction resulting in swelling and blockage of the drainage angle.")
                    st.markdown('---') 
                    st.subheader("What are the signs of glaucoma and how is it diagnosed?")
                    st.write("The most common signs noted by owners are:")
                    st.write(f"**Eye pain**. Your dog may partially close and rub at the eye. He may turn away as you touch him or pet the side of his head.")
                    st.write(f"A **watery discharge** from the eye.")
                    st.write(f"**Lethargy, loss of appetite** or even **unresponsiveness.**")
                    st.write(f"**Obvious physical swelling and bulging of the eyeball** The white of the eye (sclera) looks red and engorged.")
                    st.write(f"The cornea or clear part of the eye may become cloudy or bluish in color.")
                    st.write(f"Blindness can occur very quickly unless the increased IOP is reduced.")
                    st.write(f"**All of these signs can occur very suddenly with acute glaucoma**. In chronic glaucoma they develop more slowly. They may have been present for some time before your pet shows any signs of discomfort or clinical signs.")
                    st.write(f"Diagnosis of glaucoma depends upon accurate IOP measurement and internal eye examination using special instruments. **Acute glaucoma is an emergency**. Sometimes immediate referral to a veterinary ophthalmologist is necessary.")
                    st.markdown('---') 
                    st.subheader("What is the treatment for glaucoma?")
                    st.write("It is important to reduce the IOP as quickly as possible to reduce the risk of irreversible damage and blindness. It is also important to treat any underlying disease that may be responsible for the glaucoma. Analgesics are usually prescribed to control the pain and discomfort associated with the condition. Medications that decrease fluid production and promote drainage are often prescribed to treat the increased pressure. Long-term medical therapy may involve drugs such as carbonic anhydrase inhibitors (e.g., dorzolamide 2%, brand names Trusopt® and Cosopt®) or beta-adrenergic blocking agents (e.g., 0.5% timolol, brand names Timoptic® and Betimol®). Medical treatment often must be combined with surgery in severe or advanced cases. Veterinary ophthalmologists use various surgical techniques to reduce intraocular pressure. In some cases that do not respond to medical treatment or if blindness has developed, removal of the eye may be recommended to relieve the pain and discomfort.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/glaucoma-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hypothyroidism")
                    st.image("https://www.lifelearn-cliented.com//cms/resources/body/24023/2024_817i_thyroid_dog_5002.png")
                with col2:
                    Hypothyroidism = ''' A condition of inadequate thyroid hormone levels that leads to a reduction in a dog's metabolic state. Hypothyroidism is one of the most common hormonal (endocrine) diseases in dogs. It generally affects middle-aged dogs (average of 6–7 years of age), and it may be more common in spayed females and neutered males. A wide variety of breeds may be affected.'''
                    st.markdown(Hypothyroidism)
                with st.expander("See More Details"):
                    st.subheader("What causes hypothyroidism?")
                    st.write("In dogs, hypothyroidism is usually caused by one of two diseases: lymphocytic thyroiditis or idiopathic thyroid gland atrophy. **Lymphocytic thyroiditis** is the most common cause of hypothyroidism and is thought to be an immune-mediated disease, meaning that the immune system decides that the thyroid is abnormal or foreign and attacks it. It is unclear why this occurs; however, it is a heritable trait, so genetics plays a role. In **idiopathic thyroid gland atrophy**, normal thyroid tissue is replaced by fat tissue. This condition is also poorly understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of hypothyroidism?")
                    st.write("When the metabolic rate slows down, virtually every organ in the body is affected. Most dogs with hypothyroidism have one or more of the following signs:")
                    st.write("weight gain without an increase in appetite")
                    st.write("lethargy (tiredness) and lack of desire to exercise")
                    st.write("cold intolerance (gets cold easily)")
                    st.write("dry, dull hair with excessive shedding")
                    st.write("very thin to nearly bald hair coat")
                    st.write("increased dark pigmentation in the skin")
                    st.write("increased susceptibility and occurrence of skin and ear infections")
                    st.write("failure to re-grow hair after clipping or shaving")
                    st.write("high blood cholesterol")
                    st.write("slow heart rate")
                    st.markdown("---")
                    st.subheader("How is hypothyroidism diagnosed?")
                    st.write("The most common screening test is a total thyroxin (TT4) level. This is a measurement of the main thyroid hormone in a blood sample. A low level of TT4, along with the presence of clinical signs, is suggestive of hypothyroidism. Definitive diagnosis is made by performing a free T4 by equilibrium dialysis (free T4 by ED) or a thyroid panel that assesses the levels of multiple forms of thyroxin. If this test is low, then your dog has hypothyroidism. Some pets will have a low TT4 and normal free T4 by ED. These dogs do not have hypothyroidism. Additional tests may be necessary based on your pet's condition. See handout “Thyroid Hormone Testing in Dogs” for more information.")
                    st.markdown("---")
                    st.subheader("Can it be treated?")
                    st.write("Hypothyroidism is treatable but not curable. It is treated with oral administration of thyroid replacement hormone. This drug must be given for the rest of the dog's life. The most recommended treatment is oral synthetic thyroid hormone replacement called levothyroxine (brand names Thyro-Tabs® Canine, Synthroid®).")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hypothyroidism-in-dogs")
        
        # elif breed_label == "Airedale":          
        # elif breed_label == "Cairn":  
        # elif breed_label == "Australian Terrier": 
        # elif breed_label == "Dandie Dinmont": 
        # elif breed_label == "Boston bull": 
        # elif breed_label == "Miniature Schnauzer": 
        # elif breed_label == "Giant Schnauzer": 
        # elif breed_label == "Standard Schnauzer": 
        # elif breed_label == "Scotch Terrier": 
        # elif breed_label == "Tibetan Terrier": 
        # elif breed_label == "Silky Terrier": 
        # elif breed_label == "Soft Coated Wheaten Terrier":
        # elif breed_label == "West Highland White Terrier":  
        # elif breed_label == "Lhasa": 
        # elif breed_label == "Flat Coated Retriever":  
        # elif breed_label == "Curly Coated Retriever":  
        # elif breed_label == "Golden retriever":  
        # elif breed_label == "Labrador retriever":  
        # elif breed_label == "Chesapeake bay retriever":
        # elif breed_label == "German short haired pointer":
        # elif breed_label == "Vizsla":
        # elif breed_label == "English Setter":
        elif breed_label == "Irish Setter":
            tab1 = st.tabs(["Hypothyroidism"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hypothyroidism")
                    st.image("https://www.lifelearn-cliented.com//cms/resources/body/24023/2024_817i_thyroid_dog_5002.png")
                with col2:
                    Hypothyroidism = ''' A condition of inadequate thyroid hormone levels that leads to a reduction in a dog's metabolic state. Hypothyroidism is one of the most common hormonal (endocrine) diseases in dogs. It generally affects middle-aged dogs (average of 6–7 years of age), and it may be more common in spayed females and neutered males. A wide variety of breeds may be affected.'''
                    st.markdown(Hypothyroidism)
                with st.expander("See More Details"):
                    st.subheader("What causes hypothyroidism?")
                    st.write("In dogs, hypothyroidism is usually caused by one of two diseases: lymphocytic thyroiditis or idiopathic thyroid gland atrophy. **Lymphocytic thyroiditis** is the most common cause of hypothyroidism and is thought to be an immune-mediated disease, meaning that the immune system decides that the thyroid is abnormal or foreign and attacks it. It is unclear why this occurs; however, it is a heritable trait, so genetics plays a role. In **idiopathic thyroid gland atrophy**, normal thyroid tissue is replaced by fat tissue. This condition is also poorly understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of hypothyroidism?")
                    st.write("When the metabolic rate slows down, virtually every organ in the body is affected. Most dogs with hypothyroidism have one or more of the following signs:")
                    st.write("weight gain without an increase in appetite")
                    st.write("lethargy (tiredness) and lack of desire to exercise")
                    st.write("cold intolerance (gets cold easily)")
                    st.write("dry, dull hair with excessive shedding")
                    st.write("very thin to nearly bald hair coat")
                    st.write("increased dark pigmentation in the skin")
                    st.write("increased susceptibility and occurrence of skin and ear infections")
                    st.write("failure to re-grow hair after clipping or shaving")
                    st.write("high blood cholesterol")
                    st.write("slow heart rate")
                    st.markdown("---")
                    st.subheader("How is hypothyroidism diagnosed?")
                    st.write("The most common screening test is a total thyroxin (TT4) level. This is a measurement of the main thyroid hormone in a blood sample. A low level of TT4, along with the presence of clinical signs, is suggestive of hypothyroidism. Definitive diagnosis is made by performing a free T4 by equilibrium dialysis (free T4 by ED) or a thyroid panel that assesses the levels of multiple forms of thyroxin. If this test is low, then your dog has hypothyroidism. Some pets will have a low TT4 and normal free T4 by ED. These dogs do not have hypothyroidism. Additional tests may be necessary based on your pet's condition. See handout “Thyroid Hormone Testing in Dogs” for more information.")
                    st.markdown("---")
                    st.subheader("Can it be treated?")
                    st.write("Hypothyroidism is treatable but not curable. It is treated with oral administration of thyroid replacement hormone. This drug must be given for the rest of the dog's life. The most recommended treatment is oral synthetic thyroid hormone replacement called levothyroxine (brand names Thyro-Tabs® Canine, Synthroid®).")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hypothyroidism-in-dogs")

        elif breed_label == "Gordon Setter":
            tab1, tab2, tab3= st.tabs(["Bloat", "Cataract", "Entropion"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Bloat")
                    st.image("https://www.akc.org/wp-content/uploads/2021/09/Senior-Beagle-lying-on-a-rug-indoors.jpg")
                with col2:
                    Bloat = '''Bloat, also known as gastric dilatation-volvulus (GDV) complex, is a medical and surgical emergency. As the stomach fills with air, pressure builds, stopping blood from the hind legs and abdomen from returning to the heart. Blood pools at the back end of the body, reducing the working blood volume and sending the dog into shock.'''
                    st.markdown(Bloat)
                with st.expander("See More Details"):
                    st.subheader("What Are the Signs of Bloat in Dogs?")
                    st.write("An enlargement of the dog’s abdomen")
                    st.write("Retching")
                    st.write("Salivation")
                    st.write("Restlessness")
                    st.write("An affected dog will feel pain and might whine if you press on his belly")
                    st.write("Without treatment, in only an hour or two, your dog will likely go into shock. The heart rate will rise and the pulse will get weaker, leading to death.")
                    st.markdown("---")
                    st.subheader("Why Do Dogs Bloat?")
                    st.write("This question has perplexed veterinarians since they first identified the disease. We know air accumulates in the stomach (dilatation), and the stomach twists (the volvulus part). We don’t know if the air builds up and causes the twist, or if the stomach twists and then the air builds up.")
                    st.markdown("---")
                    st.subheader("How Is Bloat Treated?")
                    st.write("Veterinarians start by treating the shock. Once the dog is stable, he’s taken into surgery. We do two procedures. One is to deflate the stomach and turn it back to its correct position. If the stomach wall is damaged, that piece is removed. Second, because up to 90 percent of affected dogs will have this condition again, we tack the stomach to the abdominal wall (a procedure called a gastropexy) to prevent it from twisting.")
                    st.markdown("---")
                    st.subheader("How Can Bloat Be Prevented?")
                    st.write("If a dog has relatives (parents, siblings, or offspring) who have suffered from bloat, there is a higher chance he will develop bloat. These dogs should not be used for breeding.")
                    st.write("Risk of bloat is correlated to chest conformation. Dogs with a deep, narrow chest — very tall, rather than wide — suffer the most often from bloat. Great Danes, who have a high height-to-width ratio, are five-to-eight times more likely to bloat than dogs with a low height-to-width ratio.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/bloat-in-dogs/")
            with tab2:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Entropion")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Entroption = '''when the eyelids roll inward toward the eye. The fur on the eyelids and the eyelashes then rub against the surface of the eye (the cornea). This is a very painful condition that can lead to corneal ulcers.'''
                    st.markdown(Entroption)
                with st.expander("See More Details"):
                    st.write("Many Bloodhounds have abnormally large eyelids (macroblepharon) which results in an unusually large space between the eyelids.  Because of their excessive facial skin and resulting facial droop, there is commonly poor support of the outer corner of the eyelids")
                    st.markdown("---")
                    st.subheader("How is entropion treated?")
                    st.write("The treatment for entropion is surgical correction. A section of skin is removed from the affected eyelid to reverse its inward rolling. In many cases, a primary, major surgical correction will be performed, and will be followed by a second, minor corrective surgery later. Two surgeries are often performed to reduce the risk of over-correcting the entropion, resulting in an outward-rolling eyelid known as ectropion. Most dogs will not undergo surgery until they have reached their adult size at six to twelve months of age.")
                    st.markdown("---")
                    st.subheader("Should an affected dog be bred?")
                    st.write("Due to the concern of this condition being inherited, dogs with severe ectropion requiring surgical correction should not be bred.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/eyelid-entropion-in-dogs")

        elif breed_label == "Brittany Spaniel":
            tab1, tab2, tab3= st.tabs(["Cleft palate", "Cataract", "Distichiasis"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cleft Palate") 
                    st.image("https://www.bpmcdn.com/f/files/victoria/import/2021-06/25618240_web1_210605-CPL-SPCA-Lock-In-Fore-Love-Baby-Snoot-Chilliwack_4.jpg", width=200)
                with col2:
                    Cleft_Palate = '''A condition where the roof of the mouth is not closed 
                                and the inside of the nose opens into the mouth. It occurs due to a failure of the roof of the mouth to close during 
                                development in the womb. This results in a hole between the mouth and the nasal cavity. 
                                The defect can occur in the lip (primary cleft palate) or along the roof of the mouth (secondary cleft palate).
                            '''
                    st.markdown(Cleft_Palate)
                with st.expander("See More details"):
                    st.subheader("Cleft palate in puppies Prognosis")
                    st.write("A cleft palate is generally detected by visual examination of newborn puppies by the veterinary surgeon or breeder. Cleft palate of the lip or hard palate are easy to see, but soft palate defects can sometimes require sedation or general anaesthesia to visualise. Affected puppies will often have difficulty suckling and swallowing. This is often seen as coughing, gagging, and milk bubbling from the pup’s nose. In less severe defects, more subtle signs such as sneezing, snorting, failure to grow, or sudden onset of breathing difficulty (due to aspiration of milk or food) can occur.")
                    st.markdown("---")
                    st.subheader("Treatment for cleft palate in puppies")
                    st.write("Treatment depends on the severity of the condition, the age at which the diagnosis is made, and whether there are complicating factors, such as aspiration pneumonia.")
                    st.write("Small primary clefts of the lip and nostril of the dog are unlikely to cause clinical problems.")
                    st.write("Secondary cleft palates in dogs require surgical treatment to prevent long-term nasal and lung infections and to help the puppy to feed effectively. The surgery involves either creating a single flap of healthy tissue and overlapping it over the defect or creating a ‘double flap’, releasing the palate from the inside of the upper teeth, and sliding it to meet in the middle over the defect.")
                    st.markdown("---")
                    st.link_button("Source","https://www.petmd.com/dog/conditions/mouth/c_dg_cleft_palate")
            with tab2:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Distichiasis")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/2136//2023_2135i_distichia_eye_6021.jpg")
                with col2:
                    Distichiasis = '''A distichia (plural distichiae) is an extra eyelash that grows from the margin of the eyelid through the duct or opening of the meibomian gland or adjacent to it. Meibomian glands produce lubricants for the eye and their openings are located along the inside edge of the eyelids. The condition in which these abnormal eyelashes are found is called distichiasis.'''
                    st.markdown(Distichiasis)
                with st.expander("See More Details"):
                    st.subheader("What causes distichiasis?")
                    st.write("Sometimes eyelashes arise from the meibomian glands. Why the follicles develop in this abnormal location is not known, but the condition is recognized as a hereditary problem in certain breeds of dogs. Distichiasis is a rare disorder in cats.")
                    st.markdown("---")
                    st.subheader("What breeds are more likely to have distichiasis?")
                    st.write("The more commonly affected breeds include the American Cocker Spaniel, Cavalier King Charles Spaniel, Shih Tzu, Lhasa Apso, Dachshund, Shetland Sheepdog, Golden Retriever, Chesapeake Retriever, Bulldog, Boston Terrier, Pug, Boxer Dog, Maltese, and Pekingese.")
                    st.markdown("---")
                    st.subheader("How is distichiasis diagnosed?")
                    st.write("Distichiasis is usually diagnosed by identifying lashes emerging from the meibomian gland openings or by observing lashes that touch the cornea or the conjunctival lining of the affected eye. A thorough eye examination is usually necessary, including fluorescein staining of the cornea and assessment of tear production in the eyes, to assess the extent of any corneal injury and to rule out other causes of the dog's clinical signs. Some dogs will require topical anesthetics or sedatives to relieve the intense discomfort and allow a thorough examination of the tissues surrounding the eye.")
                    st.markdown("---")
                    st.subheader("How is the condition treated?")
                    st.write("Dogs that are not experiencing clinical signs with short, fine distichia may require no treatment at all. Patients with mild clinical signs may be managed conservatively, through the use of ophthalmic lubricants to protect the cornea and coat the lashes with a lubricant film. Removal of distichiae is no longer recommended, as they often grow back thicker or stiffer, but they may be removed for patients unable to undergo anesthesia or while waiting for a more permanent procedure.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/distichia-or-distichiasis-in-dogs")

        elif breed_label == "Clumber":
            tab1, tab2, tab3= st.tabs(["Ectropion", "Entropion", "Hip dysplasia"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Ectropion")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/23450//2023_1008i_Eye_ectropion_cross-section_2020-01.jpg")
                with col2:
                    Ectropion = '''Ectropion is an abnormality of the eyelids in which the eyelid (usually the lower eyelid) “rolls” outward or is everted. This causes the lower eyelids to appear droopy.'''
                    st.markdown(Ectropion)
                with st.expander("See More Details"):
                    st.subheader("What are the clinical signs of ectropion?")
                    st.write("The clinical signs are a “sagging” or “outward rolling” lower eyelid. A thick, mucoid discharge often accumulates along the eyelid margin. The eye and conjunctiva may appear reddened or inflamed. The dog may rub or paw at the eye if it becomes uncomfortable. Tears may run down the dog’s face if the medial aspect of the eyelid (the area of the eyelid toward the nose) is affected. In many cases, pigment contained in the tear fluid will cause a brownish staining of the fur beneath the eyes.")
                    st.markdown("---")
                    st.subheader("How is ectropion diagnosed?")
                    st.write("Diagnosis is usually made on physical examination. If the dog is older, blood and urine tests may be performed to search for an underlying cause for the ectropion. Corneal staining will be performed to assess the cornea and to determine if any corneal ulceration is present. Muscle or nerve biopsies may be recommended if neuromuscular disease is suspected. Testing for hypothyroidism and for antibodies against certain muscle fibers may be done if looking for underlying causes.")
                    st.markdown("---")
                    st.subheader("How is ectropion treated?")
                    st.write("The treatment for mild ectropion generally consists of medical therapy, such as lubricating eye drops and ointments to prevent the cornea and conjunctiva from drying out. Ophthalmic antibiotics may be recommended if corneal ulcers develop because of ectropion. If the condition is severe, the eyelids can be shortened surgically.")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/eyelid-ectropion-in-dogs")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Entropion")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Entroption = '''when the eyelids roll inward toward the eye. The fur on the eyelids and the eyelashes then rub against the surface of the eye (the cornea). This is a very painful condition that can lead to corneal ulcers.'''
                    st.markdown(Entroption)
                with st.expander("See More Details"):
                    st.write("Many Bloodhounds have abnormally large eyelids (macroblepharon) which results in an unusually large space between the eyelids.  Because of their excessive facial skin and resulting facial droop, there is commonly poor support of the outer corner of the eyelids")
                    st.markdown("---")
                    st.subheader("How is entropion treated?")
                    st.write("The treatment for entropion is surgical correction. A section of skin is removed from the affected eyelid to reverse its inward rolling. In many cases, a primary, major surgical correction will be performed, and will be followed by a second, minor corrective surgery later. Two surgeries are often performed to reduce the risk of over-correcting the entropion, resulting in an outward-rolling eyelid known as ectropion. Most dogs will not undergo surgery until they have reached their adult size at six to twelve months of age.")
                    st.markdown("---")
                    st.subheader("Should an affected dog be bred?")
                    st.write("Due to the concern of this condition being inherited, dogs with severe ectropion requiring surgical correction should not be bred.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/eyelid-entropion-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hip-dysplasia-in-dogs/")

        elif breed_label == "English Springer":
            tab1, tab2, tab3= st.tabs(["Cleft palate", "Distichiasis", "Epilepsy"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cleft Palate") 
                    st.image("https://www.bpmcdn.com/f/files/victoria/import/2021-06/25618240_web1_210605-CPL-SPCA-Lock-In-Fore-Love-Baby-Snoot-Chilliwack_4.jpg", width=200)
                with col2:
                    Cleft_Palate = '''A condition where the roof of the mouth is not closed 
                                and the inside of the nose opens into the mouth. It occurs due to a failure of the roof of the mouth to close during 
                                development in the womb. This results in a hole between the mouth and the nasal cavity. 
                                The defect can occur in the lip (primary cleft palate) or along the roof of the mouth (secondary cleft palate).
                            '''
                    st.markdown(Cleft_Palate)
                with st.expander("See More details"):
                    st.subheader("Cleft palate in puppies Prognosis")
                    st.write("A cleft palate is generally detected by visual examination of newborn puppies by the veterinary surgeon or breeder. Cleft palate of the lip or hard palate are easy to see, but soft palate defects can sometimes require sedation or general anaesthesia to visualise. Affected puppies will often have difficulty suckling and swallowing. This is often seen as coughing, gagging, and milk bubbling from the pup’s nose. In less severe defects, more subtle signs such as sneezing, snorting, failure to grow, or sudden onset of breathing difficulty (due to aspiration of milk or food) can occur.")
                    st.markdown("---")
                    st.subheader("Treatment for cleft palate in puppies")
                    st.write("Treatment depends on the severity of the condition, the age at which the diagnosis is made, and whether there are complicating factors, such as aspiration pneumonia.")
                    st.write("Small primary clefts of the lip and nostril of the dog are unlikely to cause clinical problems.")
                    st.write("Secondary cleft palates in dogs require surgical treatment to prevent long-term nasal and lung infections and to help the puppy to feed effectively. The surgery involves either creating a single flap of healthy tissue and overlapping it over the defect or creating a ‘double flap’, releasing the palate from the inside of the upper teeth, and sliding it to meet in the middle over the defect.")
                    st.markdown("---")
                    st.link_button("Source","https://www.petmd.com/dog/conditions/mouth/c_dg_cleft_palate")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Distichiasis")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/2136//2023_2135i_distichia_eye_6021.jpg")
                with col2:
                    Distichiasis = '''A distichia (plural distichiae) is an extra eyelash that grows from the margin of the eyelid through the duct or opening of the meibomian gland or adjacent to it. Meibomian glands produce lubricants for the eye and their openings are located along the inside edge of the eyelids. The condition in which these abnormal eyelashes are found is called distichiasis.'''
                    st.markdown(Distichiasis)
                with st.expander("See More Details"):
                    st.subheader("What causes distichiasis?")
                    st.write("Sometimes eyelashes arise from the meibomian glands. Why the follicles develop in this abnormal location is not known, but the condition is recognized as a hereditary problem in certain breeds of dogs. Distichiasis is a rare disorder in cats.")
                    st.markdown("---")
                    st.subheader("What breeds are more likely to have distichiasis?")
                    st.write("The more commonly affected breeds include the American Cocker Spaniel, Cavalier King Charles Spaniel, Shih Tzu, Lhasa Apso, Dachshund, Shetland Sheepdog, Golden Retriever, Chesapeake Retriever, Bulldog, Boston Terrier, Pug, Boxer Dog, Maltese, and Pekingese.")
                    st.markdown("---")
                    st.subheader("How is distichiasis diagnosed?")
                    st.write("Distichiasis is usually diagnosed by identifying lashes emerging from the meibomian gland openings or by observing lashes that touch the cornea or the conjunctival lining of the affected eye. A thorough eye examination is usually necessary, including fluorescein staining of the cornea and assessment of tear production in the eyes, to assess the extent of any corneal injury and to rule out other causes of the dog's clinical signs. Some dogs will require topical anesthetics or sedatives to relieve the intense discomfort and allow a thorough examination of the tissues surrounding the eye.")
                    st.markdown("---")
                    st.subheader("How is the condition treated?")
                    st.write("Dogs that are not experiencing clinical signs with short, fine distichia may require no treatment at all. Patients with mild clinical signs may be managed conservatively, through the use of ophthalmic lubricants to protect the cornea and coat the lashes with a lubricant film. Removal of distichiae is no longer recommended, as they often grow back thicker or stiffer, but they may be removed for patients unable to undergo anesthesia or while waiting for a more permanent procedure.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/distichia-or-distichiasis-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Epilepsy")
                    st.image("https://canna-pet.com/wp-content/uploads/2017/03/CP_EpilepsyDogs_1.jpg")
                with col2:
                    Epilepsy = '''A brain disorder characterized by recurrent seizures without a known cause or abnormal brain lesion (brain injury or disease). In other words, the brain appears to be normal but functions abnormally. A seizure is a sudden surge in the electrical activity of the brain causing signs such as twitching, shaking, tremors, convulsions, and/or spasms.'''
                    st.markdown(Epilepsy)
                with st.expander("See More Details"):
                    st.subheader("What Are the Symptoms of Seizures?")
                    st.write("Symptoms can include collapsing, jerking, stiffening, muscle twitching, loss of consciousness, drooling, chomping, tongue chewing, or foaming at the mouth. Dogs can fall to the side and make paddling motions with their legs. They sometimes poop or pee during the seizure. They are also not aware of their surroundings. Some dogs may look dazed, seem unsteady or confused, or stare off into space before a seizure. Afterward, your dog may be disoriented, wobbly, or temporarily blind. They may walk in circles and bump into things. They might have a lot of drool on their chin. They may try to hide.")
                    st.markdown("---")
                    st.subheader("How is epilepsy diagnosed?")
                    st.write("Epilepsy is a diagnosis of exclusion; the diagnosis of epilepsy is made only after all other causes of seizures have been ruled out. A thorough medical history and physical examination are performed, followed by diagnostic testing such as blood and urine tests and radiographs (X-rays). Additional tests such as bile acids, cerebrospinal fluid (CSF) testing, computed tomography (CT) or magnetic resonance imaging (MRI) may be recommended, depending on the initial test results. In many cases a cause is not found; these are termed idiopathic. Many epilepsy cases are grouped under this classification as the more advanced testing is often not carried out due to cost or availability. A dog’s age when seizures first start is also a prevalent factor in coming to a diagnosis.")
                    st.markdown("---")
                    st.subheader("What is the treatment of epilepsy?")
                    st.write("Anticonvulsants (anti-seizure medications) are the treatment of choice for epilepsy. There are several commonly used anticonvulsants, and once treatment is started, it will likely be continued for life. Stopping these medications suddenly can cause seizures.")
                    st.write("The risk and severity of future seizures may be worsened by stopping and re- starting anticonvulsant drugs. Therefore, anticonvulsant treatment is often only prescribed if one of the following criteria is met:")
                    st.write("**More than one seizure a month:** You will need to record the date, time, length, and severity of all episodes in order to determine medication necessity and response to treatment.")
                    st.write("**Clusters of seizures:** If your pet has groups or 'clusters' of seizures, (one seizure following another within a very short period of time), the condition may progress to status epilepticus, a life- threatening condition characterized by a constant, unending seizure that may last for hours. Status epilepticus is a medical emergency.")
                    st.write("**Grand mal or severe seizures:** Prolonged or extremely violent seizure episodes. These may worsen over time without treatment.")
                    st.markdown("---")
                    st.subheader("What is the prognosis for a pet with epilepsy?")
                    st.write("Most dogs do well on anti-seizure medication and are able to resume a normal lifestyle. Some patients continue to experience periodic break-through seizures. Many dogs require occasional medication adjustments, and some require the addition of other medications over time.")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/epilepsy-in-dogs")

        elif breed_label == "Welsh springer spaniel":
            tab1, tab2, tab3= st.tabs(["Cleft palate", "Hypothyroidism", "Hip dysplasia"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cleft Palate") 
                    st.image("https://www.bpmcdn.com/f/files/victoria/import/2021-06/25618240_web1_210605-CPL-SPCA-Lock-In-Fore-Love-Baby-Snoot-Chilliwack_4.jpg", width=200)
                with col2:
                    Cleft_Palate = '''A condition where the roof of the mouth is not closed 
                                and the inside of the nose opens into the mouth. It occurs due to a failure of the roof of the mouth to close during 
                                development in the womb. This results in a hole between the mouth and the nasal cavity. 
                                The defect can occur in the lip (primary cleft palate) or along the roof of the mouth (secondary cleft palate).
                            '''
                    st.markdown(Cleft_Palate)
                with st.expander("See More details"):
                    st.subheader("Cleft palate in puppies Prognosis")
                    st.write("A cleft palate is generally detected by visual examination of newborn puppies by the veterinary surgeon or breeder. Cleft palate of the lip or hard palate are easy to see, but soft palate defects can sometimes require sedation or general anaesthesia to visualise. Affected puppies will often have difficulty suckling and swallowing. This is often seen as coughing, gagging, and milk bubbling from the pup’s nose. In less severe defects, more subtle signs such as sneezing, snorting, failure to grow, or sudden onset of breathing difficulty (due to aspiration of milk or food) can occur.")
                    st.markdown("---")
                    st.subheader("Treatment for cleft palate in puppies")
                    st.write("Treatment depends on the severity of the condition, the age at which the diagnosis is made, and whether there are complicating factors, such as aspiration pneumonia.")
                    st.write("Small primary clefts of the lip and nostril of the dog are unlikely to cause clinical problems.")
                    st.write("Secondary cleft palates in dogs require surgical treatment to prevent long-term nasal and lung infections and to help the puppy to feed effectively. The surgery involves either creating a single flap of healthy tissue and overlapping it over the defect or creating a ‘double flap’, releasing the palate from the inside of the upper teeth, and sliding it to meet in the middle over the defect.")
                    st.markdown("---")
                    st.link_button("Source","https://www.petmd.com/dog/conditions/mouth/c_dg_cleft_palate")
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hypothyroidism")
                    st.image("https://www.lifelearn-cliented.com//cms/resources/body/24023/2024_817i_thyroid_dog_5002.png")
                with col2:
                    Hypothyroidism = ''' A condition of inadequate thyroid hormone levels that leads to a reduction in a dog's metabolic state. Hypothyroidism is one of the most common hormonal (endocrine) diseases in dogs. It generally affects middle-aged dogs (average of 6–7 years of age), and it may be more common in spayed females and neutered males. A wide variety of breeds may be affected.'''
                    st.markdown(Hypothyroidism)
                with st.expander("See More Details"):
                    st.subheader("What causes hypothyroidism?")
                    st.write("In dogs, hypothyroidism is usually caused by one of two diseases: lymphocytic thyroiditis or idiopathic thyroid gland atrophy. **Lymphocytic thyroiditis** is the most common cause of hypothyroidism and is thought to be an immune-mediated disease, meaning that the immune system decides that the thyroid is abnormal or foreign and attacks it. It is unclear why this occurs; however, it is a heritable trait, so genetics plays a role. In **idiopathic thyroid gland atrophy**, normal thyroid tissue is replaced by fat tissue. This condition is also poorly understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of hypothyroidism?")
                    st.write("When the metabolic rate slows down, virtually every organ in the body is affected. Most dogs with hypothyroidism have one or more of the following signs:")
                    st.write("weight gain without an increase in appetite")
                    st.write("lethargy (tiredness) and lack of desire to exercise")
                    st.write("cold intolerance (gets cold easily)")
                    st.write("dry, dull hair with excessive shedding")
                    st.write("very thin to nearly bald hair coat")
                    st.write("increased dark pigmentation in the skin")
                    st.write("increased susceptibility and occurrence of skin and ear infections")
                    st.write("failure to re-grow hair after clipping or shaving")
                    st.write("high blood cholesterol")
                    st.write("slow heart rate")
                    st.markdown("---")
                    st.subheader("How is hypothyroidism diagnosed?")
                    st.write("The most common screening test is a total thyroxin (TT4) level. This is a measurement of the main thyroid hormone in a blood sample. A low level of TT4, along with the presence of clinical signs, is suggestive of hypothyroidism. Definitive diagnosis is made by performing a free T4 by equilibrium dialysis (free T4 by ED) or a thyroid panel that assesses the levels of multiple forms of thyroxin. If this test is low, then your dog has hypothyroidism. Some pets will have a low TT4 and normal free T4 by ED. These dogs do not have hypothyroidism. Additional tests may be necessary based on your pet's condition. See handout “Thyroid Hormone Testing in Dogs” for more information.")
                    st.markdown("---")
                    st.subheader("Can it be treated?")
                    st.write("Hypothyroidism is treatable but not curable. It is treated with oral administration of thyroid replacement hormone. This drug must be given for the rest of the dog's life. The most recommended treatment is oral synthetic thyroid hormone replacement called levothyroxine (brand names Thyro-Tabs® Canine, Synthroid®).")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hypothyroidism-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hip dysplasia")
                    st.image("https://www.petmd.com/sites/default/files/hip_dysplasia-01_1.jpg")
                with col2:
                    Hip_dysplasia = '''A common skeletal condition, often seen in large or giant breed dogs, although it can occur in smaller breeds, as well. To understand how the condition works, owners first must understand the basic anatomy of the hip joint.'''
                    st.markdown(Hip_dysplasia)
                with st.expander("See More Details"):
                    st.subheader("What Causes Hip Dysplasia in Dogs?")
                    st.write("Several factors lead to the development of hip dysplasia in dogs, beginning with genetics. Hip dysplasia is hereditary and is especially common in larger dogs, like the Great Dane, Saint Bernard, Labrador Retriever, and German Shepherd Dog. Factors such as excessive growth rate, types of exercise, improper weight, and unbalanced nutrition can magnify this genetic predisposition.")
                    st.write("Some puppies have special nutrition requirements and need food specially formulated for large-breed puppies. These foods help prevent excessive growth, which can lead to skeletal disorders such as hip dysplasia, along with elbow dysplasia and other joint conditions. Slowing down these breeds’ growth allows their joints to develop without putting too much strain on them, helping to prevent problems down the line")
                    st.write("Improper nutrition can also influence a dog’s likelihood of developing hip dysplasia, as can giving a dog too much or too little exercise. Obesity puts a lot of stress on your dog’s joints, which can exacerbate a pre-existing condition such as hip dysplasia or even cause hip dysplasia. Talk to your vet about the best diet for your dog and the appropriate amount of exercise your dog needs each day to keep them in good physical condition.")
                    st.markdown("---")
                    st.subheader("Diagnosing Hip Dysplasia in Dogs")
                    st.write("One of the first things that your veterinarian may do is manipulate your dog’s hind legs to test the looseness of the joint. They’ll likely check for any grinding, pain, or reduced range of motion. Your dog’s physical exam may include blood work because inflammation due to joint disease can be indicated in the complete blood count. Your veterinarian will also need a history of your dog’s health and symptoms, any possible incidents or injuries that may have contributed to these symptoms, and any information you have about your dog’s parentage.")
                    st.markdown("---")
                    st.subheader("Treating Hip Dysplasia in Dogs")
                    st.write("There are quite a few treatment options for hip dysplasia in dogs, ranging from lifestyle modifications to surgery. If your dog’s hip dysplasia is not severe, or if your dog is not a candidate for surgery for medical or financial reasons, your veterinarian may recommend a nonsurgical approach.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/hip-dysplasia-in-dogs/")

        elif breed_label == "Welsh springer spaniel":
            tab1, tab2, tab3= st.tabs(["Cataract", "Hypothyroidism", "Glaucoma"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Hypothyroidism")
                    st.image("https://www.lifelearn-cliented.com//cms/resources/body/24023/2024_817i_thyroid_dog_5002.png")
                with col2:
                    Hypothyroidism = ''' A condition of inadequate thyroid hormone levels that leads to a reduction in a dog's metabolic state. Hypothyroidism is one of the most common hormonal (endocrine) diseases in dogs. It generally affects middle-aged dogs (average of 6–7 years of age), and it may be more common in spayed females and neutered males. A wide variety of breeds may be affected.'''
                    st.markdown(Hypothyroidism)
                with st.expander("See More Details"):
                    st.subheader("What causes hypothyroidism?")
                    st.write("In dogs, hypothyroidism is usually caused by one of two diseases: lymphocytic thyroiditis or idiopathic thyroid gland atrophy. **Lymphocytic thyroiditis** is the most common cause of hypothyroidism and is thought to be an immune-mediated disease, meaning that the immune system decides that the thyroid is abnormal or foreign and attacks it. It is unclear why this occurs; however, it is a heritable trait, so genetics plays a role. In **idiopathic thyroid gland atrophy**, normal thyroid tissue is replaced by fat tissue. This condition is also poorly understood.")
                    st.markdown("---")
                    st.subheader("What are the signs of hypothyroidism?")
                    st.write("When the metabolic rate slows down, virtually every organ in the body is affected. Most dogs with hypothyroidism have one or more of the following signs:")
                    st.write("weight gain without an increase in appetite")
                    st.write("lethargy (tiredness) and lack of desire to exercise")
                    st.write("cold intolerance (gets cold easily)")
                    st.write("dry, dull hair with excessive shedding")
                    st.write("very thin to nearly bald hair coat")
                    st.write("increased dark pigmentation in the skin")
                    st.write("increased susceptibility and occurrence of skin and ear infections")
                    st.write("failure to re-grow hair after clipping or shaving")
                    st.write("high blood cholesterol")
                    st.write("slow heart rate")
                    st.markdown("---")
                    st.subheader("How is hypothyroidism diagnosed?")
                    st.write("The most common screening test is a total thyroxin (TT4) level. This is a measurement of the main thyroid hormone in a blood sample. A low level of TT4, along with the presence of clinical signs, is suggestive of hypothyroidism. Definitive diagnosis is made by performing a free T4 by equilibrium dialysis (free T4 by ED) or a thyroid panel that assesses the levels of multiple forms of thyroxin. If this test is low, then your dog has hypothyroidism. Some pets will have a low TT4 and normal free T4 by ED. These dogs do not have hypothyroidism. Additional tests may be necessary based on your pet's condition. See handout “Thyroid Hormone Testing in Dogs” for more information.")
                    st.markdown("---")
                    st.subheader("Can it be treated?")
                    st.write("Hypothyroidism is treatable but not curable. It is treated with oral administration of thyroid replacement hormone. This drug must be given for the rest of the dog's life. The most recommended treatment is oral synthetic thyroid hormone replacement called levothyroxine (brand names Thyro-Tabs® Canine, Synthroid®).")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/hypothyroidism-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Glaucoma")
                    st.image("https://www.animaleyecare.com.au/images/animal-eye-care/conditions/glaucoma-in-dogs-w.jpg")
                with col2:
                    Glaucoma = '''A disease of the eye in which the pressure within the eye, called intraocular pressure (IOP), is increased. Intraocular pressure is measured using an instrument called a tonometer.'''
                    st.markdown(Glaucoma)
                with st.expander("See More Details"):
                    st.subheader("What is intraocular pressure and how is it maintained?")
                    st.write("The inside of the eyeball is filled with fluid, called aqueous humor. The size and shape of the normal eye is maintained by the amount of fluid contained within the eyeball. The pressure of the fluid inside the front or anterior chamber of the eye is known as intraocular pressure (IOP). Aqueous humor is produced by a structure called the ciliary body. In addition to producing this fluid (aqueous humor), the ciliary body contains the suspensory ligaments that hold the lens in place. Muscles in the ciliary body pull on the suspensory ligaments, controlling the shape and focusing ability of the lens.Aqueous humor contains nutrients and oxygen that are used by the structures within the eye. The ciliary body constantly produces aqueous humor, and the excess fluid is constantly drained from the eye between the cornea and the iris. This area is called the iridocorneal angle, the filtration angle, or the drainage angle.As long as the production and absorption or drainage of aqueous humor is equal, the intraocular pressure remains constant.")
                    st.markdown('---') 
                    st.subheader("What causes glaucoma?")
                    st.write("Glaucoma is caused by inadequate drainage of aqueous fluid; it is not caused by overproduction of fluid. Glaucoma is further classified as primary or secondary glaucoma.")
                    st.write(f"**Primary glaucoma** results in increased intraocular pressure in a healthy eye. Some breeds are more prone than others (see below). It occurs due to inherited anatomical abnormalities in the drainage angle.")
                    st.write(f"**Secondary glaucoma** results in increased intraocular pressure due to disease or injury to the eye. This is the most common cause of glaucoma in dogs. Causes include:")
                    st.write(f"**Uveitis** (inflammation of the interior of the eye) or severe intraocular infections, resulting in debris and scar tissue blocking the drainage angle.")
                    st.write(f"**Anterior dislocation of lens**. The lens falls forward and physically blocks the drainage angle or pupil so that fluid is trapped behind the dislocated lens.")
                    st.write(f"**Tumors** can cause physical blockage of the iridocorneal angle.")
                    st.write(f"**Intraocular bleeding.** If there is bleeding in the eye, a blood clot can prevent drainage of the aqueous humor.")
                    st.write(f"Damage to the lens. Lens proteins leaking into the eye because of a ruptured lens can cause an inflammatory reaction resulting in swelling and blockage of the drainage angle.")
                    st.markdown('---') 
                    st.subheader("What are the signs of glaucoma and how is it diagnosed?")
                    st.write("The most common signs noted by owners are:")
                    st.write(f"**Eye pain**. Your dog may partially close and rub at the eye. He may turn away as you touch him or pet the side of his head.")
                    st.write(f"A **watery discharge** from the eye.")
                    st.write(f"**Lethargy, loss of appetite** or even **unresponsiveness.**")
                    st.write(f"**Obvious physical swelling and bulging of the eyeball** The white of the eye (sclera) looks red and engorged.")
                    st.write(f"The cornea or clear part of the eye may become cloudy or bluish in color.")
                    st.write(f"Blindness can occur very quickly unless the increased IOP is reduced.")
                    st.write(f"**All of these signs can occur very suddenly with acute glaucoma**. In chronic glaucoma they develop more slowly. They may have been present for some time before your pet shows any signs of discomfort or clinical signs.")
                    st.write(f"Diagnosis of glaucoma depends upon accurate IOP measurement and internal eye examination using special instruments. **Acute glaucoma is an emergency**. Sometimes immediate referral to a veterinary ophthalmologist is necessary.")
                    st.markdown('---') 
                    st.subheader("What is the treatment for glaucoma?")
                    st.write("It is important to reduce the IOP as quickly as possible to reduce the risk of irreversible damage and blindness. It is also important to treat any underlying disease that may be responsible for the glaucoma. Analgesics are usually prescribed to control the pain and discomfort associated with the condition. Medications that decrease fluid production and promote drainage are often prescribed to treat the increased pressure. Long-term medical therapy may involve drugs such as carbonic anhydrase inhibitors (e.g., dorzolamide 2%, brand names Trusopt® and Cosopt®) or beta-adrenergic blocking agents (e.g., 0.5% timolol, brand names Timoptic® and Betimol®). Medical treatment often must be combined with surgery in severe or advanced cases. Veterinary ophthalmologists use various surgical techniques to reduce intraocular pressure. In some cases that do not respond to medical treatment or if blindness has developed, removal of the eye may be recommended to relieve the pain and discomfort.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/glaucoma-in-dogs")

        elif breed_label == "Cocker Spaniel":
            tab1, tab2, tab3= st.tabs(["Cataract", "Cleft palate", "Distichiasis"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Cleft Palate") 
                    st.image("https://www.bpmcdn.com/f/files/victoria/import/2021-06/25618240_web1_210605-CPL-SPCA-Lock-In-Fore-Love-Baby-Snoot-Chilliwack_4.jpg", width=200)
                with col2:
                    Cleft_Palate = '''A condition where the roof of the mouth is not closed 
                                and the inside of the nose opens into the mouth. It occurs due to a failure of the roof of the mouth to close during 
                                development in the womb. This results in a hole between the mouth and the nasal cavity. 
                                The defect can occur in the lip (primary cleft palate) or along the roof of the mouth (secondary cleft palate).
                            '''
                    st.markdown(Cleft_Palate)
                with st.expander("See More details"):
                    st.subheader("Cleft palate in puppies Prognosis")
                    st.write("A cleft palate is generally detected by visual examination of newborn puppies by the veterinary surgeon or breeder. Cleft palate of the lip or hard palate are easy to see, but soft palate defects can sometimes require sedation or general anaesthesia to visualise. Affected puppies will often have difficulty suckling and swallowing. This is often seen as coughing, gagging, and milk bubbling from the pup’s nose. In less severe defects, more subtle signs such as sneezing, snorting, failure to grow, or sudden onset of breathing difficulty (due to aspiration of milk or food) can occur.")
                    st.markdown("---")
                    st.subheader("Treatment for cleft palate in puppies")
                    st.write("Treatment depends on the severity of the condition, the age at which the diagnosis is made, and whether there are complicating factors, such as aspiration pneumonia.")
                    st.write("Small primary clefts of the lip and nostril of the dog are unlikely to cause clinical problems.")
                    st.write("Secondary cleft palates in dogs require surgical treatment to prevent long-term nasal and lung infections and to help the puppy to feed effectively. The surgery involves either creating a single flap of healthy tissue and overlapping it over the defect or creating a ‘double flap’, releasing the palate from the inside of the upper teeth, and sliding it to meet in the middle over the defect.")
                    st.markdown("---")
                    st.link_button("Source","https://www.petmd.com/dog/conditions/mouth/c_dg_cleft_palate")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Distichiasis")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/2136//2023_2135i_distichia_eye_6021.jpg")
                with col2:
                    Distichiasis = '''A distichia (plural distichiae) is an extra eyelash that grows from the margin of the eyelid through the duct or opening of the meibomian gland or adjacent to it. Meibomian glands produce lubricants for the eye and their openings are located along the inside edge of the eyelids. The condition in which these abnormal eyelashes are found is called distichiasis.'''
                    st.markdown(Distichiasis)
                with st.expander("See More Details"):
                    st.subheader("What causes distichiasis?")
                    st.write("Sometimes eyelashes arise from the meibomian glands. Why the follicles develop in this abnormal location is not known, but the condition is recognized as a hereditary problem in certain breeds of dogs. Distichiasis is a rare disorder in cats.")
                    st.markdown("---")
                    st.subheader("What breeds are more likely to have distichiasis?")
                    st.write("The more commonly affected breeds include the American Cocker Spaniel, Cavalier King Charles Spaniel, Shih Tzu, Lhasa Apso, Dachshund, Shetland Sheepdog, Golden Retriever, Chesapeake Retriever, Bulldog, Boston Terrier, Pug, Boxer Dog, Maltese, and Pekingese.")
                    st.markdown("---")
                    st.subheader("How is distichiasis diagnosed?")
                    st.write("Distichiasis is usually diagnosed by identifying lashes emerging from the meibomian gland openings or by observing lashes that touch the cornea or the conjunctival lining of the affected eye. A thorough eye examination is usually necessary, including fluorescein staining of the cornea and assessment of tear production in the eyes, to assess the extent of any corneal injury and to rule out other causes of the dog's clinical signs. Some dogs will require topical anesthetics or sedatives to relieve the intense discomfort and allow a thorough examination of the tissues surrounding the eye.")
                    st.markdown("---")
                    st.subheader("How is the condition treated?")
                    st.write("Dogs that are not experiencing clinical signs with short, fine distichia may require no treatment at all. Patients with mild clinical signs may be managed conservatively, through the use of ophthalmic lubricants to protect the cornea and coat the lashes with a lubricant film. Removal of distichiae is no longer recommended, as they often grow back thicker or stiffer, but they may be removed for patients unable to undergo anesthesia or while waiting for a more permanent procedure.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/distichia-or-distichiasis-in-dogs")

        elif breed_label == "Sussex Spaniel":
            tab1, tab2, tab3= st.tabs(["Cataract", "Entropion", "Distichiasis"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1: 
                    st.header("Cataract")
                    st.image("https://thumbs.dreamstime.com/z/old-age-blind-pekinese-dog-cataract-both-eyes-resting-floor-199259336.jpg")
                with col2:
                    Cataract = '''A cataract is a cloudy lens within the eye. Small cataracts are only visible with the aid of an ophthalmoscope, while large cataracts can be easy to see, as the pupil will look completely white. The larger the cataract, the more significant the loss of vision.'''
                    st.markdown(Cataract)
                with st.expander("See More Details"):
                    st.subheader("What Causes Cataracts in Dogs?")
                    st.write("Cataracts in dogs can be caused by things like nutritional impairment from the lack of arginine in a milk replacement, congenital issues where the fetus develops incorrectly, trauma or injury to the eye, or uveitis, an inflammation of the eye that results in the warping of the lens fibers. However, the two most common causes of canine cataracts are genetics and diabetes mellitus.")
                    st.markdown("---")
                    st.subheader("Signs of Cataracts in Dogs")
                    st.write("You might think vision loss would be the first sign of a cataract. But that’s not the case. Incipient cataracts, which means the cataract covers less than 15% of the lens, don’t usually affect the vision in that eye. And dogs with immature cataracts, the next stage of advancement, can still have vision, although the vision won’t be normal. The final two stages of cataracts, mature and hyper-mature cataracts, will cause vision loss in the affected eye.")
                    st.markdown("---")
                    st.subheader("Treatment for Cataracts in Dogs")
                    st.write("If a dog’s vision isn’t affected by the cataract, your veterinarian will likely advise monitoring the situation. If inflammation is present, anti-inflammatory eye drops will be prescribed to keep the eye as healthy and pain-free as possible. You’d need to continue administering these drops as long as the cataract is present.")
                    st.markdown("---")
                    st.link_button("Source","https://www.akc.org/expert-advice/health/cataracts-in-dogs-what-to-know/") 
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Entropion")
                    st.image("data:image/jpeg;base64,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")
                with col2:
                    Entroption = '''when the eyelids roll inward toward the eye. The fur on the eyelids and the eyelashes then rub against the surface of the eye (the cornea). This is a very painful condition that can lead to corneal ulcers.'''
                    st.markdown(Entroption)
                with st.expander("See More Details"):
                    st.write("Many Bloodhounds have abnormally large eyelids (macroblepharon) which results in an unusually large space between the eyelids.  Because of their excessive facial skin and resulting facial droop, there is commonly poor support of the outer corner of the eyelids")
                    st.markdown("---")
                    st.subheader("How is entropion treated?")
                    st.write("The treatment for entropion is surgical correction. A section of skin is removed from the affected eyelid to reverse its inward rolling. In many cases, a primary, major surgical correction will be performed, and will be followed by a second, minor corrective surgery later. Two surgeries are often performed to reduce the risk of over-correcting the entropion, resulting in an outward-rolling eyelid known as ectropion. Most dogs will not undergo surgery until they have reached their adult size at six to twelve months of age.")
                    st.markdown("---")
                    st.subheader("Should an affected dog be bred?")
                    st.write("Due to the concern of this condition being inherited, dogs with severe ectropion requiring surgical correction should not be bred.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/eyelid-entropion-in-dogs")
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Distichiasis")
                    st.image("https://www.lifelearn-cliented.com/cms/resources/body/2136//2023_2135i_distichia_eye_6021.jpg")
                with col2:
                    Distichiasis = '''A distichia (plural distichiae) is an extra eyelash that grows from the margin of the eyelid through the duct or opening of the meibomian gland or adjacent to it. Meibomian glands produce lubricants for the eye and their openings are located along the inside edge of the eyelids. The condition in which these abnormal eyelashes are found is called distichiasis.'''
                    st.markdown(Distichiasis)
                with st.expander("See More Details"):
                    st.subheader("What causes distichiasis?")
                    st.write("Sometimes eyelashes arise from the meibomian glands. Why the follicles develop in this abnormal location is not known, but the condition is recognized as a hereditary problem in certain breeds of dogs. Distichiasis is a rare disorder in cats.")
                    st.markdown("---")
                    st.subheader("What breeds are more likely to have distichiasis?")
                    st.write("The more commonly affected breeds include the American Cocker Spaniel, Cavalier King Charles Spaniel, Shih Tzu, Lhasa Apso, Dachshund, Shetland Sheepdog, Golden Retriever, Chesapeake Retriever, Bulldog, Boston Terrier, Pug, Boxer Dog, Maltese, and Pekingese.")
                    st.markdown("---")
                    st.subheader("How is distichiasis diagnosed?")
                    st.write("Distichiasis is usually diagnosed by identifying lashes emerging from the meibomian gland openings or by observing lashes that touch the cornea or the conjunctival lining of the affected eye. A thorough eye examination is usually necessary, including fluorescein staining of the cornea and assessment of tear production in the eyes, to assess the extent of any corneal injury and to rule out other causes of the dog's clinical signs. Some dogs will require topical anesthetics or sedatives to relieve the intense discomfort and allow a thorough examination of the tissues surrounding the eye.")
                    st.markdown("---")
                    st.subheader("How is the condition treated?")
                    st.write("Dogs that are not experiencing clinical signs with short, fine distichia may require no treatment at all. Patients with mild clinical signs may be managed conservatively, through the use of ophthalmic lubricants to protect the cornea and coat the lashes with a lubricant film. Removal of distichiae is no longer recommended, as they often grow back thicker or stiffer, but they may be removed for patients unable to undergo anesthesia or while waiting for a more permanent procedure.")
                    st.markdown("---")
                    st.link_button("Source","https://vcahospitals.com/know-your-pet/distichia-or-distichiasis-in-dogs")

        # elif breed_label == "Irish Water Spaniel":
        # elif breed_label == "Kuvasz":
        # elif breed_label == "Schipperke":
        # elif breed_label == "Groenendael":
        # elif breed_label == "Malinois":
        # elif breed_label == "Briard":
        # elif breed_label == "Kelpie":
        # elif breed_label == "Komondor":
        # elif breed_label == "Old English Sheepdog":
        # elif breed_label == "Shetland Sheepdog":
        # elif breed_label == "Collie":
        # elif breed_label == "Border Collie":
        # elif breed_label == "Bouvier Des Flandres":
        # elif breed_label == "Rottweiler":
        # elif breed_label == "German Shepherd":
        # elif breed_label == "Doberman":
        # elif breed_label == "Miniature Pinscher":
        # elif breed_label == "Greater Swiss Mountain Dog":
        # elif breed_label == "Bernese Mountain Dog":
        # elif breed_label == "Appenzeller":
        # Provide a clickable link to open Google search results
        search_query = f"{breed_label} dog images"
        search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
        link_html = f'<div style="text-align: center;"><a href="{search_url}" target="_blank" style="display: inline-block; text-align: center; cursor: pointer; color: #FF5733;">🐶 Click here to view Google search results</a></div>'
        st.markdown(link_html, unsafe_allow_html=True)
        st.markdown('---')
        
        st.markdown("<h3 style='text-align: left; color: #4d8df2; font-size: 24px;'>A group thesis project by</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; font-size: 16px;'>Kathleen L. Dabalos  |  Rosh Aubrey G. Asares  |  Ericka I. Isleta</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left; font-size: 16px;'>University of Mindanao🎓</p>", unsafe_allow_html=True)
        st.markdown('---')

def img_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

run()