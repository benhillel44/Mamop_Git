import cv2
import numpy as np

CHUNK_SIZE = 10
IMAGE_SIZE = (640, 480)

# function that returns a picture taken by the camera when a button is pressed
def take_picture():
    # open the camera
    cap = cv2.VideoCapture(0)
    # take a picture
    ret, frame = cap.read()
    # close the camera
    cap.release()
    # return the picture
    return frame


# function that takes a picture and divides it into chunks
def picture_to_chunks():
    # take a picture
    frame = take_picture()

    frame = cv2.resize(frame, IMAGE_SIZE)

    # divide the picture into chunks
    chunks = []
    for i in range(0, frame.shape[0], CHUNK_SIZE):
        for j in range(0, frame.shape[1], CHUNK_SIZE):
            # get the chunk
            chunk = frame[i:i + CHUNK_SIZE, j:j + CHUNK_SIZE]
            # do something with the chunk
            # for example, print the chunk
            # add the chunk to the list of chunks
            chunks.append(chunk)
    print(len(chunks), "chunks")

    # return the chunks
    return chunks


def chunk_to_bytes(chunk):
    # convert the chunk to bytes
    return chunk.tobytes()

def bytes_to_chunk(bytes):
    # convert the bytes to a chunk
    return np.frombuffer(bytes, dtype=np.uint8).reshape(CHUNK_SIZE, CHUNK_SIZE, 3)


def chunks_to_img(chunks):
    # put the chunks together
    chunks_per_row = IMAGE_SIZE[0] // CHUNK_SIZE
    horizontal_chunks = [cv2.hconcat([chunks[i + j] for j in range(chunks_per_row)]) for i in
                         range(0, len(chunks), chunks_per_row)]
    img = cv2.vconcat(horizontal_chunks)

    return img

def draw_image_with_partial_chunks(chunks):
    # put the chunks together
    chunks_per_row = IMAGE_SIZE[0] // CHUNK_SIZE
    chunk_amount = IMAGE_SIZE[0] * IMAGE_SIZE[1] // CHUNK_SIZE**2
    while len(chunks) < chunk_amount:
        chunks.append(np.zeros((CHUNK_SIZE, CHUNK_SIZE, 3), dtype=np.uint8))
    horizontal_chunks = [cv2.hconcat([chunks[i + j] for j in range(chunks_per_row)]) for i in
                         range(0, len(chunks), chunks_per_row)]
    img = cv2.vconcat(horizontal_chunks)

    return img


if __name__ == "__main__":
    # take a picture and divide it into chunks
    chunks = picture_to_chunks()

    byte_chunks = [chunk_to_bytes(chunk) for chunk in chunks]

    print(byte_chunks[0])
    print(len(byte_chunks[0]))

    chunks = [bytes_to_chunk(byte_chunk) for byte_chunk in byte_chunks]

    img = chunks_to_img(chunks)

    cv2.namedWindow("test")
    for i in range(0, len(chunks), 50):
        img = draw_image_with_partial_chunks(chunks[:i])
        cv2.imshow("test", img)
        cv2.waitKey(10)
    img = draw_image_with_partial_chunks(chunks)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
