import cv2
import face_recognition
import numpy as np

# Function to encode multiple images of a person
def encode_images(image_files):
    encodings = []
    for image_file in image_files:
        image = face_recognition.load_image_file(image_file)
        encodings.extend(face_recognition.face_encodings(image))
    return encodings

# Encode multiple images for each known person
modi_images = ['samples/modi.jpg']
modi_encodings = encode_images(modi_images)

trump_images = ['samples/trump.jpg']
trump_encodings = encode_images(trump_images)

erandi_images = ['samples/erandi.jpg', 'samples/erandi1.jpg', 'samples/erandi2.jpg', 'samples/erandi3.jpg']
erandi_encodings = encode_images(erandi_images)

sunimal_images = ['samples/sunimal.jpg']
sunimal_encodings = encode_images(sunimal_images)

steve_images = ['samples/steve.jpg','samples/steve1.jpg','samples/steve2.jpg','samples/steve3.jpg','samples/steve4.jpg','samples/steve5.jpg']
steve_encodings = encode_images(steve_images)

# Combine all encodings and names
known_face_encodings = modi_encodings + trump_encodings + erandi_encodings + sunimal_encodings
known_face_names = (
    ["modi"] * len(modi_encodings) +
    ["trump"] * len(trump_encodings) +
    ["erandi"] * len(erandi_encodings) +
    ["sunimal"] * len(sunimal_encodings) +
    ["Steve"] * len(steve_encodings)
)

# Test images
#'testImages/test1.jpg'
# 'testImages/test4.jpg'
test_images = ['testImages/test.jpg','testImages/test1.jpg', 'testImages/test2.jpg', 'testImages/test3.jpg','testImages/test4.webp','testImages/test4.jpg']

# Annotate and process all test images
annotated_images = []
for test_image_path in test_images:
    # Load test image
    image_to_recognize = face_recognition.load_image_file(test_image_path)

    # Detect faces and encode them
    all_face_locations = face_recognition.face_locations(image_to_recognize, model="cnn")
    all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)

    print(f'There are {len(all_face_locations)} faces in {test_image_path}.')

    # Annotate faces in the image
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        top, right, bottom, left = current_face_location

        # Check for matches
        matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding, tolerance=0.5)
        name = "Unknown Face"

        # Use the first match (or most similar match)
        if True in matches:
            match_indices = [i for i, match in enumerate(matches) if match]
            face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)
            best_match_index = match_indices[np.argmin(face_distances[match_indices])]
            name = known_face_names[best_match_index]

        # Draw a rectangle and label the face
        cv2.rectangle(image_to_recognize, (left, top), (right, bottom), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_to_recognize, name, (left, bottom + 20), font, 0.5, (255, 255, 255), 1)

    # Convert to BGR for OpenCV
    annotated_image = cv2.cvtColor(image_to_recognize, cv2.COLOR_RGB2BGR)

    # Display the annotated image
    cv2.imshow(f"Results for {test_image_path}", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
